"""RedemptionMixin: on-chain CTF position redemption via Polymarket builder relayer."""

from __future__ import annotations

import os
import time as _time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests
from web3 import Web3
from eth_abi import encode as abi_encode
from eth_account import Account as EthAccount
from eth_account.messages import encode_defunct
from py_builder_signing_sdk.signing.hmac import build_hmac_signature

if TYPE_CHECKING:
    from tracker import LiveTradeTracker

# ── On-chain redemption constants ────────────────────────────────────────────
POLYGON_RPC = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

CTF_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")

CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

PARENT_COLLECTION_ID = bytes(32)  # 0x0...0

# ── Builder relayer constants (PROXY type) ──────────────────────────────────
RELAYER_URL = "https://relayer-v2.polymarket.com"
PROXY_FACTORY = Web3.to_checksum_address("0xaB45c5A4B0c941a2F231C04C3f49182e1A254052")
RELAY_HUB = Web3.to_checksum_address("0xD216153c06E857cD7f72665E0aF1d7D82172F494")
PROXY_INIT_CODE_HASH = bytes.fromhex(
    "d21df8dc65880a8606f09fe0ce3df9b8869287ab0b058be05aa9e8af6330a00b"
)
DEFAULT_RELAY_GAS_LIMIT = 500_000


class RedemptionMixin:
    """On-chain redemption methods mixed into LiveTradeTracker."""

    def redeem_positions(
        self: "LiveTradeTracker",
        condition_id: str,
        max_retries: int = 3,
        max_poll_attempts: int = 60,
        poll_interval_s: float = 30.0,
    ) -> str | None:
        """Redeem winning CTF positions via the Polymarket builder relayer.

        First polls payoutDenominator on-chain to wait for resolution,
        then submits via the builder relayer (no POL gas needed).
        Returns the tx hash on success, None on failure.
        """
        if self.dry_run:
            print("  [REDEEM] Skipped (dry run)")
            return None

        if not condition_id:
            print("  [REDEEM] Skipped (no conditionId)")
            return None

        # Poll payoutDenominator until on-chain resolution arrives
        condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
        resolved = False
        try:
            w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
            if not w3.is_connected():
                print("  [REDEEM] ERROR: Cannot connect to Polygon RPC")
                return None
            ctf = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
            for poll in range(max_poll_attempts):
                try:
                    denom = ctf.functions.payoutDenominator(condition_bytes).call()
                    if denom > 0:
                        resolved = True
                        print(f"  [REDEEM] On-chain resolution confirmed "
                              f"(payoutDenominator={denom}, poll #{poll + 1})")
                        break
                except Exception as exc:
                    if self.debug:
                        print(f"  [REDEEM] payoutDenominator poll error: {exc}")
                if poll < max_poll_attempts - 1:
                    if poll == 0:
                        print(f"  [REDEEM] Waiting for on-chain resolution "
                              f"(polling every {poll_interval_s:.0f}s, "
                              f"up to {max_poll_attempts * poll_interval_s / 60:.0f}m)...")
                    _time.sleep(poll_interval_s)
        except Exception as exc:
            print(f"  [REDEEM] RPC connection error: {exc}")

        if not resolved:
            print(f"  [REDEEM] On-chain resolution not found after "
                  f"{max_poll_attempts * poll_interval_s / 60:.0f}m. "
                  f"Claim manually on Polymarket.")
            return None

        for attempt in range(max_retries):
            if attempt > 0:
                delay = 20 * attempt
                print(f"  [REDEEM] Retry {attempt + 1}/{max_retries} in {delay}s...")
                _time.sleep(delay)

            result = self._try_redeem_once(condition_id)
            if result is not None:
                return result

        print(f"  [REDEEM] Failed after {max_retries} tx attempts. "
              f"Claim manually on Polymarket.")
        return None

    def _try_redeem_once(self: "LiveTradeTracker", condition_id: str) -> str | None:
        """Single attempt at CTF redemption via Polymarket builder relayer (PROXY type)."""
        try:
            private_key = os.getenv("PRIVATE_KEY", "")
            if not private_key:
                print("  [REDEEM] ERROR: PRIVATE_KEY not set")
                return None
            if not private_key.startswith("0x"):
                private_key = "0x" + private_key

            builder_key = os.getenv("POLY_BUILDER_API_KEY", "")
            builder_secret = os.getenv("POLY_BUILDER_SECRET", "")
            builder_passphrase = os.getenv("POLY_BUILDER_PASSPHRASE", "")
            if not all([builder_key, builder_secret, builder_passphrase]):
                print("  [REDEEM] ERROR: Builder API creds not set "
                      "(POLY_BUILDER_API_KEY, POLY_BUILDER_SECRET, "
                      "POLY_BUILDER_PASSPHRASE)")
                return None

            signer = EthAccount.from_key(private_key)

            # 1. Encode CTF.redeemPositions(USDC, 0x0, conditionId, [1, 2])
            w3 = Web3()
            ctf = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
            condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
            redeem_data = ctf.encode_abi(
                "redeemPositions",
                args=[USDC_ADDRESS, PARENT_COLLECTION_ID, condition_bytes, [1, 2]],
            )
            if isinstance(redeem_data, str):
                redeem_bytes = bytes.fromhex(
                    redeem_data[2:] if redeem_data.startswith("0x") else redeem_data
                )
            else:
                redeem_bytes = bytes(redeem_data)

            # 2. Wrap in proxy((uint8,address,uint256,bytes)[])
            proxy_selector = w3.keccak(
                b"proxy((uint8,address,uint256,bytes)[])"
            )[:4]
            proxy_args = abi_encode(
                ["(uint8,address,uint256,bytes)[]"],
                [[(1, CTF_ADDRESS, 0, redeem_bytes)]],
            )
            proxy_data = "0x" + (proxy_selector + proxy_args).hex()

            # 3. Derive proxy wallet via CREATE2
            salt = w3.keccak(bytes.fromhex(signer.address[2:].lower()))
            create2_input = (
                b"\xff"
                + bytes.fromhex(PROXY_FACTORY[2:])
                + salt
                + PROXY_INIT_CODE_HASH
            )
            proxy_wallet = Web3.to_checksum_address(
                "0x" + w3.keccak(create2_input).hex()[-40:]
            )

            # 4. Get relay payload (nonce + relay address)
            headers = self._builder_headers(
                builder_key, builder_secret, builder_passphrase,
                "GET", f"/relay-payload?address={signer.address}&type=PROXY",
            )
            resp = requests.get(
                f"{RELAYER_URL}/relay-payload",
                params={"address": signer.address, "type": "PROXY"},
                headers=headers,
            )
            resp.raise_for_status()
            payload = resp.json()
            relay_address = payload["address"]
            nonce = payload["nonce"]

            # 5. Create struct hash and sign (EIP-191)
            gas_limit = DEFAULT_RELAY_GAS_LIMIT
            struct_data = (
                b"rlx:"
                + bytes.fromhex(signer.address[2:])
                + bytes.fromhex(PROXY_FACTORY[2:])
                + bytes.fromhex(proxy_data[2:])
                + int(0).to_bytes(32, "big")
                + int(0).to_bytes(32, "big")
                + int(gas_limit).to_bytes(32, "big")
                + int(nonce).to_bytes(32, "big")
                + bytes.fromhex(RELAY_HUB[2:])
                + bytes.fromhex(relay_address.replace("0x", ""))
            )
            struct_hash = w3.keccak(struct_data)
            msg = encode_defunct(struct_hash)
            sig = EthAccount.sign_message(msg, private_key)
            signature = "0x" + sig.signature.hex()

            # 6. POST /submit
            body = {
                "type": "PROXY",
                "from": signer.address,
                "to": PROXY_FACTORY,
                "proxyWallet": proxy_wallet,
                "data": proxy_data,
                "nonce": str(nonce),
                "signature": signature,
                "signatureParams": {
                    "gasPrice": "0",
                    "gasLimit": str(gas_limit),
                    "relayerFee": "0",
                    "relayHub": RELAY_HUB,
                    "relay": relay_address,
                },
                "metadata": "Redeem winnings",
            }

            print(f"  [REDEEM] Submitting via relayer "
                  f"(conditionId={condition_id[:10]}..., "
                  f"proxy={proxy_wallet[:10]}...)...")

            submit_headers = self._builder_headers(
                builder_key, builder_secret, builder_passphrase,
                "POST", "/submit", body,
            )
            submit_headers["Content-Type"] = "application/json"
            resp = requests.post(
                f"{RELAYER_URL}/submit",
                json=body,
                headers=submit_headers,
            )
            resp.raise_for_status()
            result = resp.json()
            if self.debug:
                print(f"  [REDEEM] Relayer response: {result}")
            tx_id = result.get("transactionID") or result.get("transactionId", "")
            tx_hash = result.get("transactionHash", "")

            print(f"  [REDEEM] Relayer accepted: id={tx_id}, hash={tx_hash}")

            if not tx_id:
                print(f"  [REDEEM] WARNING: Relayer returned no transaction ID. "
                      f"Raw response: {result}")
                return tx_hash if tx_hash else None

            print(f"  [REDEEM] Waiting for confirmation...")

            # 7. Poll /transaction until confirmed
            confirmed = False
            for poll in range(30):
                _time.sleep(2)
                poll_headers = self._builder_headers(
                    builder_key, builder_secret, builder_passphrase,
                    "GET", f"/transaction?id={tx_id}",
                )
                try:
                    r = requests.get(
                        f"{RELAYER_URL}/transaction",
                        params={"id": tx_id},
                        headers=poll_headers,
                    )
                    r.raise_for_status()
                    tx_data = r.json()
                    state = ""
                    if isinstance(tx_data, list) and tx_data:
                        state = tx_data[0].get("state", "")
                        tx_hash = tx_data[0].get("transactionHash", tx_hash)
                    elif isinstance(tx_data, dict):
                        state = tx_data.get("state", "")
                        tx_hash = tx_data.get("transactionHash", tx_hash)

                    if state in ("STATE_MINED", "STATE_CONFIRMED"):
                        confirmed = True
                        print(f"  [REDEEM] Confirmed! (state={state}, "
                              f"hash={tx_hash})")
                        break
                    elif state == "STATE_FAILED":
                        print(f"  [REDEEM] Relayer tx FAILED (id={tx_id})")
                        break
                except Exception as poll_exc:
                    if self.debug:
                        print(f"  [REDEEM] Poll error: {poll_exc}")

            if confirmed:
                self._log({
                    "type": "redemption",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tx_hash": tx_hash,
                    "tx_id": tx_id,
                    "condition_id": condition_id,
                    "status": "success",
                })
                return tx_hash
            else:
                print(f"  [REDEEM] Relayer tx not confirmed after 60s "
                      f"(id={tx_id}, hash={tx_hash})")
                self._log({
                    "type": "redemption",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tx_hash": tx_hash,
                    "tx_id": tx_id,
                    "condition_id": condition_id,
                    "status": "relayer_timeout",
                })
                return None

        except Exception as exc:
            print(f"  [REDEEM] ERROR: {type(exc).__name__}: {exc}")
            self._log({
                "type": "redemption",
                "ts": datetime.now(timezone.utc).isoformat(),
                "condition_id": condition_id,
                "status": "error",
                "error": str(exc),
            })
            return None

    @staticmethod
    def _builder_headers(key: str, secret: str, passphrase: str,
                         method: str, path: str, body=None) -> dict:
        """Generate HMAC auth headers for the Polymarket builder relayer."""
        ts = str(int(_time.time()))
        sig = build_hmac_signature(secret, ts, method, path, body)
        return {
            "POLY_BUILDER_API_KEY": key,
            "POLY_BUILDER_PASSPHRASE": passphrase,
            "POLY_BUILDER_SIGNATURE": sig,
            "POLY_BUILDER_TIMESTAMP": ts,
        }
