# yapf: disable
# The MIT License (MIT)
# Copyright © 2023 OpenTensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import os
import argparse

import pandas as pd
import bittensor
from typing import List, Optional
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
from rich.console import Text
from tqdm import tqdm
from substrateinterface.exceptions import SubstrateRequestException
from .utils import get_delegates_details, DelegatesDetails
from . import defaults

import os
import bittensor
from typing import List, Dict, Optional

from bittensor.optimal_staking import optimal_staking

def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


console = bittensor.__console__


# Uses rich console to pretty print a table of delegates.
def show_delegates(
    delegates: List["bittensor.DelegateInfo"],
    prev_delegates: Optional[List["bittensor.DelegateInfo"]],
    width: Optional[int] = None,
    do_print: bool = True,
    topk: int = 10,
    filter_below: float = None,
    show_registration: bool = False,
):
    """Pretty prints a table of delegates sorted by total stake."""
    delegates.sort(key=lambda delegate: delegate.total_stake, reverse=True)
    prev_delegates_dict = {}
    if prev_delegates is not None:
        for prev_delegate in prev_delegates:
            prev_delegates_dict[prev_delegate.hotkey_ss58] = prev_delegate

    registered_delegate_info: Optional[
        Dict[str, DelegatesDetails]
    ] = get_delegates_details(url=bittensor.__delegates_details_url__)
    if registered_delegate_info is None:
        bittensor.__console__.print(
            ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
        )
        registered_delegate_info = {}

    table = Table(show_footer=True, width=width, pad_edge=False, box=None, expand=True)
    table.add_column(
        "[overline white]INDEX",
        str(len(delegates)),
        footer_style="overline white",
        style="bold white",
    )
    table.add_column(
        "[overline white]DELEGATE",
        style="rgb(50,163,219)",
        no_wrap=True,
        justify="left",
    )
    table.add_column(
        "[overline white]SS58",
        str(len(delegates)),
        footer_style="overline white",
        style="bold yellow",
    )
    table.add_column(
        "[overline white]NOMINATORS", justify="center", style="green", no_wrap=True
    )
    table.add_column(
        "[overline white]DELEGATE STAKE(\u03C4)", justify="right", no_wrap=True
    )
    table.add_column(
        "[overline white]TOTAL STAKE(\u03C4)",
        justify="right",
        style="green",
        no_wrap=True,
    )
    table.add_column("[overline white]CHANGE/(4h)", style="grey0", justify="center")
    if show_registration:
        table.add_column(
            "[overline white]SUBNETS", justify="right", style="white", no_wrap=False
        )
        table.add_column("[overline white]VPERMIT", justify="right", no_wrap=False)
    # table.add_column("[overline white]TAKE", style='white', no_wrap=True)
    table.add_column("[overline white]TOTAL REWARD", style='green', justify='center')
    table.add_column(
        "[overline white]NOMINATOR/(24h)/k\u03C4", style="green", justify="center"
    )
    table.add_column("[overline white]DELEGATE/(24h)", style="green", justify="center")
    table.add_column("[overline white]Desc", style="rgb(50,163,219)")
    # table.add_column("[overline white]DESCRIPTION", style='white')

    rewards = []
    for i, delegate in enumerate(delegates):
        if delegate.total_daily_return.tao == 0:
            continue
        if filter_below is not None and delegate.total_stake.tao < filter_below:
            continue
        owner_stake = next(
            map(
                lambda x: x[1],  # get stake
                filter(
                    lambda x: x[0] == delegate.owner_ss58, delegate.nominators
                ),  # filter for owner
            ),
            bittensor.Balance.from_rao(0),  # default to 0 if no owner stake.
        )
        if delegate.hotkey_ss58 in registered_delegate_info:
            delegate_name = registered_delegate_info[delegate.hotkey_ss58].name
            delegate_url = registered_delegate_info[delegate.hotkey_ss58].url
            delegate_description = registered_delegate_info[
                delegate.hotkey_ss58
            ].description
        else:
            delegate_name = ""
            delegate_url = ""
            delegate_description = ""

        if delegate.hotkey_ss58 in prev_delegates_dict:
            prev_stake = prev_delegates_dict[delegate.hotkey_ss58].total_stake
            if prev_stake == 0:
                rate_change_in_stake_str = "[green]100%[/green]"
            else:
                rate_change_in_stake = (
                    100
                    * (float(delegate.total_stake) - float(prev_stake))
                    / float(prev_stake)
                )
                if rate_change_in_stake > 0:
                    rate_change_in_stake_str = "[green]{:.2f}%[/green]".format(
                        rate_change_in_stake
                    )
                elif rate_change_in_stake < 0:
                    rate_change_in_stake_str = "[red]{:.2f}%[/red]".format(
                        rate_change_in_stake
                    )
                else:
                    rate_change_in_stake_str = "[grey0]0%[/grey0]"
        else:
            rate_change_in_stake_str = "[grey0]NA[/grey0]"

        reward = bittensor.Balance.from_tao( delegate.total_daily_return.tao * (1000/ ( 0.001 + delegate.total_stake.tao ) ))
        raw_row_data = [
            i,
            delegate.hotkey_ss58,
            delegate.total_stake.tao,
            delegate.total_daily_return.tao,
            reward,
        ]
        registration_info = []
        if show_registration:
            registration_info = [
                str(delegate.registrations),
                str(
                    [
                        "*" if subnet in delegate.validator_permits else ""
                        for subnet in delegate.registrations
                    ]
                ),
            ]
        row = [
            str(i),
            Text(delegate_name, style=f"link {delegate_url}"),
            f"{delegate.hotkey_ss58:8.8}...",
            str(len([nom for nom in delegate.nominators if nom[1].rao > 0])),
            f"{owner_stake!s:13.13}",
            f"{delegate.total_stake!s:13.13}",
            rate_change_in_stake_str,
            *registration_info,
            f"{delegate.total_daily_return.tao!s:6.6}",
            # f'{delegate.take * 100:.1f}%',
            f'{reward!s:6.6}',
            f"{bittensor.Balance.from_tao( delegate.total_daily_return.tao * (0.18) ) !s:6.6}",
            str(delegate_description)
            # f'{delegate_profile.description:140.140}',
        ]
        table.add_row(*row)
        rewards.append((i, row, raw_row_data))

    table.add_row('best', *['-' * 8] * (len(table.columns) - 1))
    top_rewards_row = sorted(rewards, key=lambda x: x[-1][-1], reverse=True)[:topk]
    if do_print:
        top_rewards_row.append(rewards[0])  # add bittensor org.
    for r in top_rewards_row:
        table.add_row(*r[1])
    if do_print:
        bittensor.__console__.print(table)

    if do_print:
        for r in top_rewards_row:
            bittensor.__console__.print(r[-1][0], r[-1][1], r[-1][-1])

    return top_rewards_row


class DelegateStakeCommand:
    @staticmethod
    def run(cli):
        """Delegates stake to a chain delegate."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)
        if isinstance(config.get("delegate_ss58key"), list):
            for ss58, amount in zip(config.get("delegate_ss58key"), config.get('amount')):
                if ss58 == config.get('delegate_ss58key')[-1]:
                    # stake all remaining balance at the end
                    amount = subtensor.get_balance(wallet.coldkeypub.ss58_address)
                    config.stake_all = True
                console.print("Stake \u03C4{} to {}".format(amount, ss58))
                subtensor.delegate(
                    wallet=wallet,
                    delegate_ss58=ss58,
                    amount=amount,
                    wait_for_inclusion=True,
                    prompt=not config.no_prompt
                )
        else:
            subtensor.delegate(
                wallet=wallet,
                delegate_ss58=config.get("delegate_ss58key"),
                amount=config.get("amount"),
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        delegate_stake_parser = parser.add_parser(
            "delegate", help="""Delegate Stake to an account."""
        )
        delegate_stake_parser.add_argument(
            "--delegate_ss58key",
            "--delegate_ss58",
            dest="delegate_ss58key",
            type=str,
            required=False,
            help="""The ss58 address of the choosen delegate""",
        )
        delegate_stake_parser.add_argument(
            "--all", dest="stake_all", action="store_true"
        )
        delegate_stake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        delegate_stake_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        delegate_stake_parser.add_argument(
            '--config_file',
            type=str,
            required=False,
            default=None,
        )
        bittensor.wallet.add_args(delegate_stake_parser)
        bittensor.subtensor.add_args(delegate_stake_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if config.get('config_file') is not None:
            df = pd.read_csv(config.get('config_file'))
            config.delegate_ss58key = df['ss58'].to_list()
            config.amount = df['amount'].to_list()
        if not config.get("delegate_ss58key"):
            # Check for delegates.
            with bittensor.__console__.status(":satellite: Loading delegates..."):
                subtensor = bittensor.subtensor(config=config)
                delegates: List[bittensor.DelegateInfo] = subtensor.get_delegates()
                try:
                    prev_delegates = subtensor.get_delegates(
                        max(0, subtensor.block - 1200)
                    )
                except SubstrateRequestException:
                    prev_delegates = None

            if prev_delegates is None:
                bittensor.__console__.print(
                    ":warning: [yellow]Could not fetch delegates history[/yellow]"
                )

            if len(delegates) == 0:
                console.print(
                    ":cross_mark: [red]There are no delegates on {}[/red]".format(
                        subtensor.network
                    )
                )
                sys.exit(1)

            delegates.sort(key=lambda delegate: delegate.total_stake, reverse=True)
            show_delegates(delegates, prev_delegates=prev_delegates)
            delegate_index = Prompt.ask("Enter delegate index")
            config.delegate_ss58key = str(delegates[int(delegate_index)].hotkey_ss58)
            console.print(
                "Selected: [yellow]{}[/yellow]".format(config.delegate_ss58key)
            )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Get amount.
        if not config.get("amount") and not config.get("stake_all"):
            if not Confirm.ask(
                "Stake all Tao from account: [bold]'{}'[/bold]?".format(
                    config.wallet.get("name", defaults.wallet.name)
                )
            ):
                amount = Prompt.ask("Enter Tao amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark: [red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.stake_all = True


class DelegateUnstakeCommand:
    @staticmethod
    def run(cli):
        """Undelegates stake from a chain delegate."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)
        for ss58 in config.get("delegate_ss58key"):
            subtensor.undelegate(
                wallet=wallet,
                delegate_ss58=ss58,
                amount=config.get("amount"),
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        undelegate_stake_parser = parser.add_parser(
            "undelegate", help="""Undelegate Stake from an account."""
        )
        undelegate_stake_parser.add_argument(
            "--entirely", action="store_true"
        )
        undelegate_stake_parser.add_argument(
            "--delegate_ss58key",
            "--delegate_ss58",
            dest="delegate_ss58key",
            type=str,
            required=False,
            help="""The ss58 address of the choosen delegate""",
        )
        undelegate_stake_parser.add_argument(
            "--all", dest="unstake_all", action="store_true"
        )
        undelegate_stake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        bittensor.wallet.add_args(undelegate_stake_parser)
        bittensor.subtensor.add_args(undelegate_stake_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.get("delegate_ss58key"):
            subtensor = bittensor.subtensor( config = config )
            if config.get('entirely'):
                wallets = _get_coldkey_wallets_for_path( config.wallet.path )
                addrs = []
                for wallet in tqdm(wallets):
                    if not wallet.coldkeypub_file.exists_on_device(): continue
                    delegates = subtensor.get_delegated( coldkey_ss58=wallet.coldkeypub.ss58_address )
                    for delegate in delegates:
                        for coldkey_addr, staked in delegate[0].nominators:
                            if coldkey_addr == wallet.coldkeypub.ss58_address and staked.tao > 0:
                                addrs.append(str(delegate[0].hotkey_ss58))
                config.delegate_ss58key = addrs
                config.unstake_all = True
            else:
                # Check for delegates.
                with bittensor.__console__.status(":satellite: Loading delegates..."):
                    delegates: List[bittensor.DelegateInfo] = subtensor.get_delegates()
                    try:
                        prev_delegates = subtensor.get_delegates(
                            max(0, subtensor.block - 1200)
                        )
                    except SubstrateRequestException:
                        prev_delegates = None

                if prev_delegates is None:
                    bittensor.__console__.print(
                        ":warning: [yellow]Could not fetch delegates history[/yellow]"
                    )

                if len(delegates) == 0:
                    console.print(
                        ":cross_mark: [red]There are no delegates on {}[/red]".format(
                            subtensor.network
                        )
                    )
                    sys.exit(1)

                delegates.sort(key=lambda delegate: delegate.total_stake, reverse=True)
                show_delegates(delegates, prev_delegates=prev_delegates)
                delegate_index = Prompt.ask("Enter delegate index")
                config.delegate_ss58key = [str(delegates[int(delegate_index)].hotkey_ss58)]
                console.print(
                    "Selected: [yellow]{}[/yellow]".format(config.delegate_ss58key)
                )

        # Get amount.
        if not config.get("amount") and not config.get("unstake_all"):
            if not Confirm.ask(
                "Unstake all Tao to account: [bold]'{}'[/bold]?".format(
                    config.wallet.get("name", defaults.wallet.name)
                )
            ):
                amount = Prompt.ask("Enter Tao amount to unstake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark: [red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.unstake_all = True


class ListDelegatesCommand:
    @staticmethod
    def run(cli):
        r"""
        List all delegates on the network.
        """
        subtensor = bittensor.subtensor(config=cli.config)
        with bittensor.__console__.status(":satellite: Loading delegates..."):
            delegates: bittensor.DelegateInfo = subtensor.get_delegates()
            try:
                prev_delegates = subtensor.get_delegates(max(0, subtensor.block - 1200))
            except SubstrateRequestException:
                prev_delegates = None

        if prev_delegates is None:
            bittensor.__console__.print(
                ":warning: [yellow]Could not fetch delegates history[/yellow]"
            )

        show_delegates(
            delegates,
            prev_delegates=prev_delegates,
            width=cli.config.get("width", None),
            show_registration=cli.config.get("show_registration", False),
            topk=20,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_delegates_parser = parser.add_parser(
            "list_delegates", help="""List all delegates on the network"""
        )
        list_delegates_parser.add_argument(
            "--show_registration",
            help="""Show registration info""",
            action="store_true",
        )
        bittensor.subtensor.add_args(list_delegates_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass


class NominateCommand:
    @staticmethod
    def run(cli):
        r"""Nominate wallet."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor = bittensor.subtensor(config=cli.config)

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        # Check if the hotkey is already a delegate.
        if subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
            bittensor.__console__.print(
                "Aborting: Hotkey {} is already a delegate.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        result: bool = subtensor.nominate(wallet)
        if not result:
            bittensor.__console__.print(
                "Could not became a delegate on [white]{}[/white]".format(
                    subtensor.network
                )
            )
        else:
            # Check if we are a delegate.
            is_delegate: bool = subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address)
            if not is_delegate:
                bittensor.__console__.print(
                    "Could not became a delegate on [white]{}[/white]".format(
                        subtensor.network
                    )
                )
                return
            bittensor.__console__.print(
                "Successfully became a delegate on [white]{}[/white]".format(
                    subtensor.network
                )
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        nominate_parser = parser.add_parser(
            "nominate", help="""Become a delegate on the network"""
        )
        bittensor.wallet.add_args(nominate_parser)
        bittensor.subtensor.add_args(nominate_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class MyDelegatesCommand:
    @staticmethod
    def run(cli):
        """Delegates stake to a chain delegate."""
        config = cli.config.copy()
        if config.get("all", d=None) == True:
            wallets = _get_coldkey_wallets_for_path(config.wallet.path)
        else:
            wallets = [bittensor.wallet(config=config)]
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)

        show_registration = config.get("show_registration", False)

        table = Table(show_footer=True, pad_edge=False, box=None, expand=True)
        table.add_column(
            "[overline white]Wallet", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]OWNER",
            style="rgb(50,163,219)",
            no_wrap=True,
            justify="left",
        )
        table.add_column(
            "[overline white]SS58", footer_style="overline white", style="bold yellow"
        )
        table.add_column(
            "[overline green]Delegation",
            footer_style="overline green",
            style="bold green",
        )
        table.add_column(
            "[overline green]\u03C4/24h",
            footer_style="overline green",
            style="bold green",
        )
        table.add_column(
            "[overline white]NOMS", justify="center", style="green", no_wrap=True
        )
        table.add_column(
            "[overline white]OWNER STAKE(\u03C4)", justify="right", no_wrap=True
        )
        table.add_column(
            "[overline white]TOTAL STAKE(\u03C4)",
            justify="right",
            style="green",
            no_wrap=True,
        )
        if show_registration:
            table.add_column(
                "[overline white]SUBNETS", justify="right", style="white", no_wrap=False
            )
            table.add_column("[overline white]VPERMIT", justify="right", no_wrap=False)
        table.add_column("[overline white]TOTAL RETURN", style='green', justify='center')
        table.add_column("[overline white]24h/k\u03C4", style="green", justify="center")
        table.add_column("[overline white]Desc", style="rgb(50,163,219)")
        total_delegated = 0

        for wallet in tqdm(wallets):
            if not wallet.coldkeypub_file.exists_on_device():
                continue
            delegates = subtensor.get_delegated(
                coldkey_ss58=wallet.coldkeypub.ss58_address
            )

            my_delegates = {}  # hotkey, amount
            for delegate in delegates:
                for coldkey_addr, staked in delegate[0].nominators:
                    if (
                        coldkey_addr == wallet.coldkeypub.ss58_address
                        and staked.tao > 0
                    ):
                        my_delegates[delegate[0].hotkey_ss58] = staked

            delegates.sort(key=lambda delegate: delegate[0].total_stake, reverse=True)
            total_delegated += sum(my_delegates.values())

            registered_delegate_info: Optional[
                DelegatesDetails
            ] = get_delegates_details(url=bittensor.__delegates_details_url__)
            if registered_delegate_info is None:
                bittensor.__console__.print(
                    ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
                )
                registered_delegate_info = {}

            total_delegate = 0
            total_emission = 0
            for i, delegate in enumerate(delegates):
                owner_stake = next(
                    map(
                        lambda x: x[1],  # get stake
                        filter(
                            lambda x: x[0] == delegate[0].owner_ss58,
                            delegate[0].nominators,
                        ),  # filter for owner
                    ),
                    bittensor.Balance.from_rao(0),  # default to 0 if no owner stake.
                )
                if delegate[0].hotkey_ss58 in registered_delegate_info:
                    delegate_name = registered_delegate_info[
                        delegate[0].hotkey_ss58
                    ].name
                    delegate_url = registered_delegate_info[delegate[0].hotkey_ss58].url
                    delegate_description = registered_delegate_info[
                        delegate[0].hotkey_ss58
                    ].description
                else:
                    delegate_name = ""
                    delegate_url = ""
                    delegate_description = ""

                if delegate[0].hotkey_ss58 in my_delegates:
                    delegate_amount = my_delegates[delegate[0].hotkey_ss58]
                    emission = delegate[0].total_daily_return.tao * (delegate_amount/delegate[0].total_stake.tao)
                    registration_info = []
                    if show_registration:
                        registration_info = [
                            str(delegate[0].registrations),
                            str(
                                [
                                    "*"
                                    if subnet in delegate[0].validator_permits
                                    else ""
                                    for subnet in delegate[0].registrations
                                ]
                            ),
                        ]
                    table.add_row(
                        wallet.name,
                        Text(delegate_name, style=f"link {delegate_url}"),
                        f"{delegate[0].hotkey_ss58:8.8}...",
                        f"{delegate_amount!s:13.13}",
                        f"{emission!s:6.6}",
                        str(len(delegate[0].nominators)),
                        f"{owner_stake!s:13.13}",
                        f"{delegate[0].total_stake!s:13.13}",
                        *registration_info,
                        # f'{delegate.take * 100:.1f}%',s
                        f"{delegate[0].total_daily_return.tao!s:6.6}",
                        f"{ delegate[0].total_daily_return.tao * ( 1000 / ( 0.001 + delegate[0].total_stake.tao ) )!s:6.6}",
                        str(delegate_description)
                        # f'{delegate_profile.description:140.140}',
                    )
                    total_delegate += delegate_amount
                    total_emission += emission
            table.add_row(*["-" * 8] * len(table.columns))  # empty line
            table.add_row(
                'total',
                '',
                '',
                f'{total_delegate!s:13.13}',
                f'{total_emission!s:6.6}',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                ''
            )
            table.add_row(
                'after tax',
                '',
                '',
                '',
                f'{total_emission * 0.82!s:6.6}',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                ''
            )

        bittensor.__console__.print(table)
        bittensor.__console__.print("Total delegated Tao: {}".format(total_delegated))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        delegate_stake_parser = parser.add_parser(
            "my_delegates",
            help="""Show all delegates where I am delegating a positive amount of stake""",
        )
        delegate_stake_parser.add_argument(
            "--all",
            action="store_true",
            help="""Check all coldkey wallets.""",
            default=False,
        )
        delegate_stake_parser.add_argument(
            "--show_registration",
            help="""Show registration info""",
            action="store_true",
        )
        bittensor.wallet.add_args(delegate_stake_parser)
        bittensor.subtensor.add_args(delegate_stake_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
            not config.get("all", d=None)
            and not config.is_set("wallet.name")
            and not config.no_prompt
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)



class OptimalDelegatesCommand:

    @staticmethod
    def run( cli ):
        '''Compute Optimal delegate amount for top-k accounts.'''
        config = cli.config.copy()
        wallets = _get_coldkey_wallets_for_path( config.wallet.path )
        config.topk = sorted([int(x) for x in config.topk.split(',')])

        # list part
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )
        with bittensor.__console__.status(":satellite: Loading delegates..."):
            delegates: bittensor.DelegateInfo = subtensor.get_delegates()
            try:
                prev_delegates = subtensor.get_delegates(max(0, subtensor.block - 1200))
            except SubstrateRequestException:
                prev_delegates = None

        if prev_delegates is None:
            bittensor.__console__.print(
                ":warning: [yellow]Could not fetch delegates history[/yellow]"
            )

        my_delegates = {} # hotkey, amount
        total_balance = 0
        for wallet in tqdm(wallets):
            if not wallet.coldkeypub_file.exists_on_device(): continue
            wallet_delegates = subtensor.get_delegated( coldkey_ss58=wallet.coldkeypub.ss58_address )
            cold_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
            total_balance += cold_balance.tao

            for delegate in wallet_delegates:
                for coldkey_addr, staked in delegate[0].nominators:
                    if coldkey_addr == wallet.coldkeypub.ss58_address and staked.tao > 0:
                        if delegate[0].hotkey_ss58 in my_delegates:
                            my_delegates[ delegate[0].hotkey_ss58 ] += staked.tao
                        else:
                            my_delegates[ delegate[0].hotkey_ss58 ] = staked.tao
                        total_balance += staked.tao

        delegate_info = []
        largest_k = config.topk[-1]
        top_candidates = show_delegates(delegates, prev_delegates = prev_delegates, width = cli.config.get('width', None),
                                        topk=largest_k, do_print=False, filter_below=cli.config.get("filter_below"))

        amount = config.amount if config.amount > 0 else total_balance

        daily_returns = []
        other_stakes = []
        addresses = []
        indices = []
        for candidate in tqdm(top_candidates, desc='topks'):
            index, hotkey_ss58, total_stake, total_daily_return, _ = candidate[-1]
            addresses.append(hotkey_ss58)
            daily_returns.append(total_daily_return)
            indices.append(index)
            if hotkey_ss58 in my_delegates:
                other_stakes.append(total_stake - my_delegates[hotkey_ss58])
            else:
                other_stakes.append(total_stake)

            stake_amount, estimated_reward = optimal_staking( amount, daily_returns, other_stakes, steps=config.steps, lr=config.lr )

            delegate_info.append((indices, addresses, other_stakes, daily_returns, stake_amount, estimated_reward))

        table = Table(show_footer=True, pad_edge=False, box=None, expand=True)
        table.add_column("[overline white]Wallet", footer_style = "overline white", style='bold white')
        table.add_column("[overline white]INDEX", footer_style = "overline white", style='bold white')
        table.add_column("[overline white]SS58", footer_style = "overline white", style='bold yellow')
        table.add_column("[overline white]Total Stake", footer_style = "overline white", style='bold yellow')
        table.add_column("[overline white]Daily Return", footer_style = "overline white", style='bold yellow')
        table.add_column("[overline green]Optimal Del. Amnt", footer_style = "overline green", style='bold green')
        table.add_column("[overline green]\u03C4/24h", footer_style = "overline green", style='bold green')

        best_idx = max(range(len(delegate_info)), key=lambda i: sum(delegate_info[i][-1]))
        indices, addresses, total_stakes, daily_returns, stake_amount, estimated_reward = delegate_info[best_idx]
        for idx, addr, stake, returns, amnt, reward in sorted(zip(indices, addresses, total_stakes, daily_returns, stake_amount, estimated_reward), key=lambda x: x[0]):
            table.add_row(wallet.name,
                          str(idx),
                          f'{addr}',
                          f'{stake}',
                          f'{returns}',
                          f'{amnt!s:13.13}',
                          f'{reward!s:6.6}')

        table.add_row(*["-" * 8] * len(table.columns))  # empty line
        table.add_row(
            'total',
            '',
            '',
            '',
            '',
            f'{sum(stake_amount)!s:13.13}',
            f'{sum(estimated_reward)!s:6.6}',
        )

        bittensor.__console__.print(table)
        if config.output_csv is not None:
            with open(config.output_csv, 'w') as f:
                f.write('ss58,amount\n')
                for addr, amnt in zip(addresses, stake_amount):
                    f.write(f'{addr},{amnt}\n')
            bittensor.__console__.print(f"Wrote config to {config.output_csv}")




    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        delegate_stake_parser = parser.add_parser(
            'optimal_delegates',
            help='''Compute Optimal Delegate Amount'''
        )
        delegate_stake_parser.add_argument(
            '--topk',
            type=str,
            default='10',
            required=False,
            help='''top-k yielding delegate accounts'''
        )
        delegate_stake_parser.add_argument(
            '--amount',
            type=float,
            default=0,
            required=False,
            help='''Amount in Tao to distribute'''
        )
        delegate_stake_parser.add_argument(
            '--steps',
            type=int,
            default=500,
            required=False,
            help='''Number of optimization steps'''
        )
        delegate_stake_parser.add_argument(
            '--lr',
            type=float,
            default=1,
            required=False,
            help='''learning rate'''
        )
        delegate_stake_parser.add_argument(
            '--output_csv',
            type=str,
            default=None,
            required=False,
            help='''output file to save the config'''
        )
        delegate_stake_parser.add_argument(
            '--filter_below',
            type=float,
            default=None,
            required=False,
            help='''filter delegates with total stake below this threshold'''
        )
        delegate_stake_parser.add_argument(
            '--no_version_checking',
            action='store_true',
            help='''Set false to stop cli version checking''',
            default = False
        )
        bittensor.wallet.add_args( delegate_stake_parser )
        bittensor.subtensor.add_args( delegate_stake_parser )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        pass
