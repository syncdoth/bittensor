import argparse

import torch


def reward_func(x, total_daily_return, other_stake=0.5, is_owner=False):
    my_reward = total_daily_return * (x / (0.001 + x + other_stake))
    if is_owner:
        my_reward += total_daily_return * (other_stake / (0.001 + x + other_stake)) * 0.18
    return my_reward


def optimal_staking(
    my_balance: float,
    daily_returns: list[float],
    other_stakes: list[float],
    steps=100,
    lr=1,
    return_loss=False,
    verbose=False,
):
    """Find optimal staking amount using SGD."""
    assert len(daily_returns) == len(other_stakes)
    if not isinstance(daily_returns, torch.Tensor):
        daily_returns = torch.FloatTensor(daily_returns)
    if not isinstance(other_stakes, torch.Tensor):
        other_stakes = torch.FloatTensor(other_stakes)

    # ML part
    W = torch.nn.Parameter(torch.full_like(daily_returns, my_balance / len(daily_returns)))
    optimizer = torch.optim.SGD([W], lr=lr)

    loss_hist = []
    grad_hist = []
    for s in range(steps):
        M = reward_func(W, daily_returns, other_stake=other_stakes)
        if verbose and s == 0:
            print("initial flat distribution total reward:")
            print(M.sum().item())

        loss = daily_returns.sum() - M.sum()  # daily_returns.sum() is upper-bound

        loss.backward()
        grad_hist.append(W.grad.norm().item())
        loss_hist.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        W.data = W.data * my_balance / W.data.sum()  # rescale

    stake_amounts = W.tolist()
    estimated_rewards = reward_func(W, daily_returns, other_stake=other_stakes).tolist()

    outputs = (stake_amounts, estimated_rewards)
    if return_loss:
        outputs += (loss_hist, grad_hist)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--balance', type=float, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    R = [1.0358, 0.9874, 0.8914, 1.1965, 1.0647, 0.9398, 1.0668, 845.44]
    O = [
        0.596777046, 0.876512084, 0.629875825, 1.328150578, 0.437902847, 1.459060901, 1.782424745,
        767725.6832
    ]

    amount, rewards, loss_hist, grad_hist = optimal_staking(args.balance,
                                                            R,
                                                            O,
                                                            steps=args.steps,
                                                            verbose=args.verbose,
                                                            return_loss=True)

    print()
    print("upper bound:", sum(R))
    print("loss:", loss_hist[-1])
    print("grad:", grad_hist[-1])
    print()
    print("stake:", amount)
    print("reward:", rewards)
    print()
    print("total tao earned / 24hr:", sum(rewards))


if __name__ == '__main__':
    main()
