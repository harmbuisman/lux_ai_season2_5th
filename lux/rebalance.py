from lux.actions import Water, Rebase, Metal
from lux.utils import lprint
from lux.constants import TTMAX, MIN_DONOR_WATER_FACTORY, IS_KAGGLE
from lux.unit import Unit
from typing import List


def rebalance(self, units: List[Unit], verbose=False):
    for unit in units:
        if haul_metal(self, unit, verbose=verbose):
            continue

        if haul_water(self, unit, verbose=verbose):
            continue

        if rebase_unit(self, unit, verbose=verbose):
            continue


def handle_low_water(self, units: List[Unit], verbose=False):
    low_water_factories = [
        f
        for f in self.factories
        if (
            f.out_of_water_time() < TTMAX
            or (f.available_water() < 50 and len(f.ice_hubs) == 0)
        )
    ]
    if verbose:
        lprint(
            f"low water factories: {[(f, f.available_water()) for f in low_water_factories]}"
        )
    if not low_water_factories:
        return

    high_water_factories = [
        f
        for f in self.factories
        if f.available_water() > MIN_DONOR_WATER_FACTORY
        and f.out_of_water_time() >= TTMAX
    ]

    if verbose:
        lprint(
            f"high water factories: {[(f, f.available_water()) for f in high_water_factories]}"
        )

    if not high_water_factories:
        return

    for low_water_factory in sorted(
        low_water_factories, key=lambda f: f.available_water()
    ):
        for unit in sorted(
            [
                u
                for u in units
                if u.is_light
                and u.factory in high_water_factories
                and u.near_my_factory()
            ],
            key=lambda u: (
                u.factory.center.distance_to(low_water_factory.center),
                u.point.distance_to(u.factory),
            ),
        ):
            if unit.factory.available_water() > MIN_DONOR_WATER_FACTORY:
                water = Water(
                    unit, destinations=low_water_factory.edge_tiles(no_chargers=True)
                )
                if water.execute():
                    if not low_water_factories:
                        break


def haul_water(self, unit: Unit, verbose=False):
    """Haul water to low water factories, non urgent case"""
    if unit.is_heavy:
        if verbose:
            lprint(f"heavy unit {unit.id} not hauling water")
        return False

    if not unit.near_my_factory():
        if verbose:
            lprint(f"unit {unit.id} not near factory")
        return False

    factory = unit.factory
    if factory.available_water() < 300:
        if verbose:
            lprint(f"factory {factory.id} has enough water {factory.available_water()}")

        return False

    low_water_factories = [
        f
        for f in self.factories
        if f.available_water() < 125 and f.distance_to(factory) < 20
    ]

    if not low_water_factories:
        low_water_factories = [
            f
            for f in self.factories
            if f.available_water() < 250 and f.distance_to(factory) < 20
        ]
        if not low_water_factories:
            if verbose:
                lprint(
                    f"No low or medium water factories: {[(f, f.available_water()) for f in self.factories]}"
                )
            return False

    closest_low_water_factory = sorted(low_water_factories, key=factory.distance_to)[0]

    water = Water(
        unit, destinations=closest_low_water_factory.edge_tiles(no_chargers=True)
    )
    return water.execute()


def haul_metal(self, unit: Unit, verbose=False):
    if unit.is_heavy:
        return False

    if not unit.near_my_factory():
        return False

    if unit.available_power(include_factory=True) < 0.5 * unit.battery_capacity:
        return False

    factory = unit.factory

    low_metal_factory = Metal.get_closest_low_metal_factory(unit)
    high_metal_factories = Metal.get_high_metal_factories(unit)

    if verbose:
        lprint(
            f"unit {unit.id} low_metal_factory {low_metal_factory} high_metal_factories {high_metal_factories}"
        )
    if low_metal_factory and factory in high_metal_factories:
        start_charge = unit.power < 0.95 * unit.battery_capacity
        if unit.required_from_to(
            unit.point, low_metal_factory.center
        ) > unit.available_power(start_charge):
            if verbose:
                lprint(
                    f"Not enough power to haul metal to {low_metal_factory}: avPWR:{unit.available_power(start_charge)}"
                )
            return False
        metal = Metal(
            unit,
            destinations=low_metal_factory.edge_tiles(no_chargers=True),
            start_charge=start_charge,
        )
        return metal.execute()
    return False


def handle_no_heavy_factories(self, units: List[Unit]):
    no_heavy_factories = [
        f
        for f in self.factories
        # when changing this in future, also see episode 50081537  around 149 and len(f.ice_hubs) > 0  #
        if f.n_heavies == 0 and f.available_water() < 50
    ]
    if not no_heavy_factories:
        return

    for factory in no_heavy_factories:
        for unit in sorted(units, key=lambda u: u.point.distance_to(factory)):
            if not unit.is_heavy:
                continue
            if (
                unit.point.distance_to(factory) > 20
                and unit.point.closest_factory.is_own
            ):
                continue
            if unit.factory.n_heavies > 1:
                rebase = Rebase(unit, destinations=factory.edge_tiles(no_chargers=True))
                if rebase.execute():
                    break


def rebase_unit(self, unit, verbose=False):
    # if factory has no unit, rebase units
    no_unit_factories = [f for f in self.factories if not f.units]
    max_units = max(len(f.units) for f in self.factories)
    max_unit_factories = [f for f in self.factories if len(f.units) == max_units]

    if not unit.point.closest_factory.is_own:
        return

    if no_unit_factories and max_units > 2 and unit.factory in max_unit_factories:
        rebase = Rebase(unit, destinations=no_unit_factories[0].edge_tiles())
        if rebase.execute():
            return True

    factory = unit.factory
    if (
        factory.n_heavies > 1
        and len(factory.ice_hubs) == 0
        and factory.n_power_hub_positions < factory.n_heavies
        and factory.available_power() < 750
    ):
        nearby_factories = [
            f
            for f in self.factories
            if f.distance_to(factory) < 15
            and f.available_power() > 500 + 50 * f.distance_to(factory)
            and f != factory
        ]
        if nearby_factories:
            destinations = sorted(
                nearby_factories, key=lambda f: f.distance_to(factory)
            )[0].edge_tiles(no_chargers=True)
            if unit.point in destinations:
                assert IS_KAGGLE, "unit already at destination"
                return False
            rebase = Rebase(unit, destinations=destinations)
            if rebase.execute():
                return True
    return False
