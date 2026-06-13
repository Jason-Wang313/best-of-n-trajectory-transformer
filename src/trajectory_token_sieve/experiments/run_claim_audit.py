from __future__ import annotations

from ..audit import write_claim_audit


def main() -> None:
    result = write_claim_audit()
    print(result["verdict"])


if __name__ == "__main__":
    main()
