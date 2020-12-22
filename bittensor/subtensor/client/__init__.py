from bittensor.subtensor.interface import SubstrateWSInterface, Keypair
import netaddr
from loguru import logger


class WSClient:
    custom_type_registry = {
        "runtime_id": 2,
        "types": {
            "NeuronMetadataOf": {
                "type": "struct",
                "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"], ["uid", "u64"], ["coldkey", "AccountId"]]
            }
        }
    }


    def __init__(self, socket : str, keypair: Keypair):
        host, port = socket.split(":")

        self.substrate = SubstrateWSInterface(
            host=host,
            port=int(port),
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry=self.custom_type_registry,
        )

        self.__keypair = keypair

    '''
    PRIVATE METHODS
    '''

    def __int_to_ip(self, int_val):
        return str(netaddr.IPAddress(int_val))

    def __ip_to_int(self, str_val):
        return int(netaddr.IPAddress(str_val))


    '''
    PUBLIC METHODS
    '''

    def connect(self):
        logger.trace("connect() C")
        self.substrate.connect()

    def is_connected(self):
        return self.substrate.is_connected()

    async def subscribe(self, ip: str, port: int, coldkey: str):
        params = {
            'ip': self.__ip_to_int(ip),
            'port': port, 
            'ip_type': 4,
            'coldkey': coldkey
        }

        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)  # Waiting for inclusion and other does not work

    async def unsubscribe(self, keypair=None):
        if not keypair:
            keypair = self.__keypair

        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='unsubscribe'
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)

    async def get_peers(self):
        peers = await self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Peers')

        return peers

    async def get_balance(self, address):
        result  = await self.substrate.get_runtime_state(
            module='System',
            storage_function='Account',
            params=[address],
            block_hash=None
        )

        balance_info = result.get('result')

        return balance_info['data']['free']

    async def add_stake(self, amount, keypair):
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='add_stake',
            call_params={'stake_amount': amount}
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)

    async def set_weights(self, destinations, values, keypair, wait_for_inclusion=False):
        call = await self.substrate.compose_call(
            call_module = 'SubtensorModule',
            call_function = 'set_weights',
            call_params = {'dests': destinations, 'weights': values}
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=wait_for_inclusion)

    async def emit(self, destinations, values, keypair, wait_for_inclusion=False):
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='set_weights',
            call_params = {'dests': destinations, 'weights': values}
        )
        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=keypair)
        await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=wait_for_inclusion)

    async def get_current_block(self):
        return await self.substrate.get_block_number(None)

    async def neurons(self, pubkey=None):
        if pubkey:
            result = await self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[pubkey]
            )
            return result['result']

        else:
            neurons = await self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            return neurons

    async def get_stake_for_uid(self, uid):
        stake = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Stake',
            params = [uid]
        )
        return stake['result']

    async def weight_uids_for_uid(self, uid):
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightUids',
            params = [uid]
        )
        return result['result']

    async def weight_vals_for_uid(self, uid):
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightVals',
            params = [uid]
        )
        return result['result']

    async def get_last_emit_data_for_uid(self, uid):
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='LastEmit',
            params = [uid]
        )
        return result['result']
        
    async def get_last_emit_data(self):
        result = await self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        return result


