"""Small bookkeeping demo for the Stage I retention signal.

This is NOT an RL experiment. It simply demonstrates what the score means.
"""

from retention_rl.retention.metrics import capability_gap, retention_deficit


def main() -> None:
    insert_nll = 1.20
    current_nll = 2.05

    historical_return = 900.0
    current_revisit_return = 650.0

    f_score = retention_deficit(insert_nll, current_nll)
    gap = capability_gap(historical_return, current_revisit_return)

    print("Retention-aware RL: Stage I signal demo")
    print("-----------------------------------------")
    print(f"NLL at VEM insertion : {insert_nll:.3f}")
    print(f"Current NLL          : {current_nll:.3f}")
    print(f"Retention deficit F  : {f_score:.3f}")
    print(f"Historical return    : {historical_return:.1f}")
    print(f"Revisit mean return  : {current_revisit_return:.1f}")
    print(f"Capability gap G     : {gap:.1f}")


if __name__ == "__main__":
    main()
