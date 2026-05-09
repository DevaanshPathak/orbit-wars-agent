import math
from collections import namedtuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 0.35
ROTATION_RADIUS_LIMIT = 50.0
MAX_FLEET_SPEED = 6.0

CAPTURE_BUFFER = 3
LEDGER_HORIZON = 120
DEFENSE_HORIZON = 42
MAX_ETA = 115
MAX_MOVES = 14
OPENING_END_STEP = 80
ENDGAME_STEP = 405
EVACUATE_HORIZON = 14

Candidate = namedtuple(
    "Candidate",
    ["kind", "score", "target_id", "parts", "eta", "ships", "reason"],
)

_OBS_CACHE = {}
_CURRENT_STATE = None


def obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _prepare_obs_cache(obs):
    global _OBS_CACHE
    planets = [Planet(*p) for p in obs_get(obs, "planets", [])]
    fleets = [Fleet(*f) for f in obs_get(obs, "fleets", [])]
    initial_planets = [Planet(*p) for p in obs_get(obs, "initial_planets", [])]
    _OBS_CACHE = {
        "obs_id": id(obs),
        "planets": planets,
        "planet_by_id": {p.id: p for p in planets},
        "fleets": fleets,
        "initial_by_id": {p.id: p for p in initial_planets},
        "comet_ids": set(obs_get(obs, "comet_planet_ids", []) or []),
    }


def _cache_for(obs):
    if _OBS_CACHE.get("obs_id") == id(obs):
        return _OBS_CACHE
    return None


def as_planets(obs):
    cache = _cache_for(obs)
    if cache is not None:
        return cache["planets"]
    return [Planet(*p) for p in obs_get(obs, "planets", [])]


def as_fleets(obs):
    cache = _cache_for(obs)
    if cache is not None:
        return cache["fleets"]
    return [Fleet(*f) for f in obs_get(obs, "fleets", [])]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_segment_distance(point, start, end):
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return dist(point, start)
    t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (start[0] + t * vx, start[1] + t * vy)
    return dist(point, closest)


def swept_pair_hit(fleet_old, fleet_new, planet_old, planet_new, radius):
    rel_x = fleet_old[0] - planet_old[0]
    rel_y = fleet_old[1] - planet_old[1]
    vel_x = (fleet_new[0] - fleet_old[0]) - (planet_new[0] - planet_old[0])
    vel_y = (fleet_new[1] - fleet_old[1]) - (planet_new[1] - planet_old[1])
    a = vel_x * vel_x + vel_y * vel_y
    b = 2.0 * (rel_x * vel_x + rel_y * vel_y)
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def path_crosses_sun(start_xy, end_xy):
    return (
        point_to_segment_distance((CENTER, CENTER), start_xy, end_xy)
        < SUN_RADIUS + SUN_MARGIN
    )


def compute_fleet_speed(num_ships):
    ships = max(1, int(num_ships))
    speed = 1.0 + (MAX_FLEET_SPEED - 1.0) * (
        math.log(ships) / math.log(1000)
    ) ** 1.5
    return min(speed, MAX_FLEET_SPEED)


def _planet_by_id(obs, planet_id):
    cache = _cache_for(obs)
    if cache is not None:
        return cache["planet_by_id"].get(planet_id)
    return next((planet for planet in as_planets(obs) if planet.id == planet_id), None)


def _initial_planet_by_id(obs, planet_id):
    cache = _cache_for(obs)
    if cache is not None:
        return cache["initial_by_id"].get(planet_id)
    return next(
        (
            Planet(*raw)
            for raw in obs_get(obs, "initial_planets", [])
            if raw[0] == planet_id
        ),
        None,
    )


def _position_step_index(step):
    # The environment exposes initial positions at agent steps 0 and 1.
    return max(0.0, float(step) - 1.0)


def predict_planet_position(planet, current_step, future_step, obs):
    cache = _cache_for(obs)
    comet_ids = cache["comet_ids"] if cache is not None else set(
        obs_get(obs, "comet_planet_ids", []) or []
    )
    if planet.id in comet_ids:
        pos = predict_comet_position(planet.id, current_step, future_step, obs)
        return pos if pos is not None else (planet.x, planet.y)

    initial = _initial_planet_by_id(obs, planet.id)
    if initial is None:
        return (planet.x, planet.y)

    orbital_radius = dist((initial.x, initial.y), (CENTER, CENTER))
    if orbital_radius + planet.radius >= ROTATION_RADIUS_LIMIT:
        return (planet.x, planet.y)

    angle0 = math.atan2(initial.y - CENTER, initial.x - CENTER)
    angle = angle0 + obs_get(obs, "angular_velocity", 0.0) * _position_step_index(
        future_step
    )
    return (
        CENTER + orbital_radius * math.cos(angle),
        CENTER + orbital_radius * math.sin(angle),
    )


def predict_comet_position(comet_planet_id, current_step, future_step, obs):
    delta = int(round(float(future_step) - float(current_step)))
    for group in obs_get(obs, "comets", []) or []:
        planet_ids = group.get("planet_ids", [])
        if comet_planet_id not in planet_ids:
            continue
        path_index = planet_ids.index(comet_planet_id)
        paths = group.get("paths", [])
        if path_index >= len(paths):
            return None
        idx = int(group.get("path_index", -1)) + delta
        path = paths[path_index]
        if idx < 0 or idx >= len(path):
            return None
        return (float(path[idx][0]), float(path[idx][1]))
    return None


def _comet_path_data(comet_planet_id, obs):
    for group in obs_get(obs, "comets", []) or []:
        planet_ids = group.get("planet_ids", [])
        if comet_planet_id not in planet_ids:
            continue
        path_index = planet_ids.index(comet_planet_id)
        paths = group.get("paths", [])
        if path_index >= len(paths):
            return None
        return group, paths[path_index]
    return None


def _first_visible_comet_offset(comet_planet_id, current_step, obs):
    del current_step
    data = _comet_path_data(comet_planet_id, obs)
    if data is None:
        return None
    group, path = data
    idx = int(group.get("path_index", -1))
    if idx >= len(path):
        return None
    # The first placement from the off-board placeholder is not collidable.
    if idx < 0:
        return 2 if len(path) > 1 else None
    return 1


def _planet_path_for_offset(planet, current_step, obs, offset):
    if offset < 1:
        return None

    cache = _cache_for(obs)
    comet_ids = cache["comet_ids"] if cache is not None else set(
        obs_get(obs, "comet_planet_ids", []) or []
    )
    if planet.id in comet_ids:
        data = _comet_path_data(planet.id, obs)
        if data is None:
            return None
        group, path = data
        current_idx = int(group.get("path_index", -1))
        old_idx = current_idx + offset - 1
        new_idx = current_idx + offset
        if old_idx >= len(path):
            return None
        if old_idx < 0:
            old_pos = (planet.x, planet.y)
            check = old_pos[0] >= 0.0 and old_pos[1] >= 0.0
        else:
            old_pos = (float(path[old_idx][0]), float(path[old_idx][1]))
            check = True
        if new_idx >= len(path):
            new_pos = old_pos
        elif new_idx < 0:
            new_pos = (planet.x, planet.y)
        else:
            new_pos = (float(path[new_idx][0]), float(path[new_idx][1]))
        return old_pos, new_pos, check

    old_pos = predict_planet_position(
        planet, current_step, current_step + offset - 1, obs
    )
    new_pos = predict_planet_position(planet, current_step, current_step + offset, obs)
    return old_pos, new_pos, True


def _first_collision_for_moving_fleet(owner, start_xy, angle, ships, state, horizon):
    del owner
    speed = compute_fleet_speed(ships)
    old_xy = start_xy
    ux = math.cos(angle)
    uy = math.sin(angle)
    for turn in range(1, int(horizon) + 1):
        new_xy = (old_xy[0] + ux * speed, old_xy[1] + uy * speed)
        for planet in state.planets:
            paths = getattr(state, "planet_path_cache", {}).get(planet.id)
            if paths is not None:
                path = paths[turn] if turn < len(paths) else None
            else:
                path = _planet_path_for_offset(planet, state.step, state.obs, turn)
            if path is None:
                continue
            planet_old, planet_new, check = path
            if check and swept_pair_hit(
                old_xy, new_xy, planet_old, planet_new, planet.radius
            ):
                return planet.id, turn

        if not (0.0 <= new_xy[0] <= BOARD_SIZE and 0.0 <= new_xy[1] <= BOARD_SIZE):
            return None, turn
        if point_to_segment_distance((CENTER, CENTER), old_xy, new_xy) < SUN_RADIUS:
            return None, turn
        old_xy = new_xy
    return None, None


def first_collision_for_launch(source, angle, ships, state, horizon=MAX_ETA):
    start_xy = (
        source.x + math.cos(angle) * (source.radius + 0.1),
        source.y + math.sin(angle) * (source.radius + 0.1),
    )
    return _first_collision_for_moving_fleet(
        source.owner, start_xy, angle, ships, state, horizon
    )


def _first_collision_for_existing_fleet(fleet, state, horizon=LEDGER_HORIZON):
    return _first_collision_for_moving_fleet(
        fleet.owner, (fleet.x, fleet.y), fleet.angle, fleet.ships, state, horizon
    )


def _target_position(target_planet_id, current_step, future_step, obs):
    planet = _planet_by_id(obs, target_planet_id)
    if planet is None:
        return None
    cache = _cache_for(obs)
    comet_ids = cache["comet_ids"] if cache is not None else set(
        obs_get(obs, "comet_planet_ids", []) or []
    )
    if planet.id in comet_ids:
        return predict_comet_position(planet.id, current_step, future_step, obs)
    return predict_planet_position(planet, current_step, future_step, obs)


def _sun_safe_angle(source_xy, target_xy, target_radius):
    direct_angle = math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])
    if not path_crosses_sun(source_xy, target_xy):
        return direct_angle, target_xy

    base_dist = dist(source_xy, target_xy)
    if base_dist <= 1e-9:
        return None

    # Aim at the target disk's visible edge instead of nudging blindly.
    for mult in (0.55, 0.8, 1.05, 1.3):
        for sign in (1.0, -1.0):
            perp = direct_angle + sign * math.pi / 2.0
            aim = (
                target_xy[0] + math.cos(perp) * target_radius * mult,
                target_xy[1] + math.sin(perp) * target_radius * mult,
            )
            if not path_crosses_sun(source_xy, aim):
                return math.atan2(aim[1] - source_xy[1], aim[0] - source_xy[0]), aim

    return None


def solve_intercept_angle(source_xy, target_planet_id, num_ships, current_step, obs):
    target = _planet_by_id(obs, target_planet_id)
    if target is None:
        return None

    speed = compute_fleet_speed(num_ships)
    target_xy = _target_position(target_planet_id, current_step, current_step, obs)
    if target_xy is None:
        first_offset = _first_visible_comet_offset(target_planet_id, current_step, obs)
        if first_offset is None:
            return None
        target_xy = _target_position(
            target_planet_id, current_step, current_step + first_offset, obs
        )
        if target_xy is None:
            return None

    eta = max(1.0, dist(source_xy, target_xy) / speed)
    if target.id in (obs_get(obs, "comet_planet_ids", []) or []):
        first_offset = _first_visible_comet_offset(target_planet_id, current_step, obs)
        if first_offset is not None:
            eta = max(eta, float(first_offset))
    for _ in range(5):
        target_xy = _target_position(
            target_planet_id, current_step, current_step + eta, obs
        )
        if target_xy is None:
            return None
        eta = max(1.0, dist(source_xy, target_xy) / speed)

    safe = _sun_safe_angle(source_xy, target_xy, target.radius)
    if safe is None:
        return None
    angle, aim_xy = safe
    eta = max(1.0, dist(source_xy, aim_xy) / speed)
    if eta > MAX_ETA:
        return None
    return angle, eta


def _fleet_arrival_eta(fleet, target_planet, current_step, obs, max_turns=MAX_ETA):
    speed = compute_fleet_speed(fleet.ships)
    ux = math.cos(fleet.angle)
    uy = math.sin(fleet.angle)
    start = (fleet.x, fleet.y)

    target_xy = predict_planet_position(target_planet, current_step, current_step, obs)
    eta = max(0.1, dist(start, target_xy) / speed)
    for _ in range(4):
        target_xy = predict_planet_position(
            target_planet, current_step, current_step + eta, obs
        )
        rel_x = target_xy[0] - start[0]
        rel_y = target_xy[1] - start[1]
        along = rel_x * ux + rel_y * uy
        if along < -target_planet.radius:
            return None
        eta = max(0.1, along / speed)

    if eta > max_turns:
        return None
    target_xy = predict_planet_position(target_planet, current_step, current_step + eta, obs)
    fleet_xy = (start[0] + ux * speed * eta, start[1] + uy * speed * eta)
    if not (0.0 <= fleet_xy[0] <= BOARD_SIZE and 0.0 <= fleet_xy[1] <= BOARD_SIZE):
        return None
    if path_crosses_sun(start, fleet_xy):
        return None
    if dist(fleet_xy, target_xy) <= target_planet.radius + max(0.6, speed * 0.45):
        return eta
    return None


def _resolve_combat(owner, garrison, arrivals):
    if not arrivals:
        return owner, garrison

    sorted_forces = sorted(arrivals.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_forces[0]
    survivor_owner = top_owner
    survivor_ships = top_ships
    if len(sorted_forces) > 1:
        second_ships = sorted_forces[1][1]
        if top_ships == second_ships:
            survivor_ships = 0
        else:
            survivor_ships = top_ships - second_ships

    if survivor_ships <= 0:
        return owner, garrison
    if survivor_owner == owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


class ArrivalLedger:
    def __init__(self, state):
        self.state = state
        self.arrivals = {planet.id: {} for planet in state.planets}
        self._timelines = {}
        self._build_existing_arrivals()

    def _build_existing_arrivals(self):
        for fleet in self.state.fleets:
            planet_id, turn = _first_collision_for_existing_fleet(
                fleet, self.state, LEDGER_HORIZON
            )
            if planet_id is None or turn is None or turn > LEDGER_HORIZON:
                continue
            by_turn = self.arrivals.setdefault(planet_id, {})
            by_owner = by_turn.setdefault(turn, {})
            by_owner[fleet.owner] = by_owner.get(fleet.owner, 0) + fleet.ships

    def timeline(self, planet_id):
        if planet_id in self._timelines:
            return self._timelines[planet_id]

        planet = self.state.planet_by_id.get(planet_id)
        if planet is None:
            return []

        owner = planet.owner
        garrison = float(planet.ships)
        result = [(owner, garrison)]
        arrivals_by_turn = self.arrivals.get(planet_id, {})
        for turn in range(1, LEDGER_HORIZON + 1):
            if owner != -1:
                garrison += planet.production
            owner, garrison = _resolve_combat(
                owner, garrison, arrivals_by_turn.get(turn, {})
            )
            result.append((owner, garrison))

        self._timelines[planet_id] = result
        return result

    def state_at(self, planet_id, turn):
        timeline = self.timeline(planet_id)
        if not timeline:
            return -1, 0.0
        idx = max(0, min(LEDGER_HORIZON, int(math.ceil(turn))))
        return timeline[idx]

    def first_not_owned_turn(self, planet_id, owner, horizon=DEFENSE_HORIZON):
        timeline = self.timeline(planet_id)
        for turn in range(1, min(horizon, len(timeline) - 1) + 1):
            if timeline[turn][0] != owner:
                return turn
        return None

    def needed_to_capture(self, planet_id, eta, attacker):
        turn = max(1, int(math.ceil(eta)))
        owner, garrison = self.state_at(planet_id, turn)
        if owner == attacker:
            return 0

        planet = self.state.planet_by_id.get(planet_id)
        if planet is None:
            return 10**9

        buffer = CAPTURE_BUFFER
        if owner != -1:
            buffer += 2
        if planet.production >= 4:
            buffer += 1
        if self.state.step > ENDGAME_STEP:
            buffer = max(1, buffer - 2)
        return max(1, int(math.ceil(garrison + buffer)))


class GameState:
    def __init__(self, obs):
        _prepare_obs_cache(obs)
        self.obs = obs
        self.step = int(obs_get(obs, "step", 0))
        self.player = obs_get(obs, "player", 0)
        self.planets = as_planets(obs)
        self.fleets = as_fleets(obs)
        cache = _cache_for(obs)
        self.planet_by_id = cache["planet_by_id"]
        self.initial_by_id = cache["initial_by_id"]
        self.comet_ids = cache["comet_ids"]

        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner not in (-1, self.player)]
        self.neutral_planets = [
            p for p in self.planets if p.owner == -1 and p.id not in self.comet_ids
        ]
        self.comet_planets = [p for p in self.planets if p.id in self.comet_ids]
        self.planet_path_cache = self._build_planet_path_cache(
            max(LEDGER_HORIZON, MAX_ETA) + 2
        )
        self.ledger = ArrivalLedger(self)
        self.totals = {
            owner: self._total_ships(owner)
            for owner in range(4)
        }
        self.my_total = self.totals.get(self.player, 0)
        self.enemy_total = sum(
            total for owner, total in self.totals.items() if owner != self.player
        )
        self.my_production = sum(p.production for p in self.my_planets)
        self.enemy_production = sum(p.production for p in self.enemy_planets)
        self.enemy_players = [
            owner for owner, total in self.totals.items() if owner != self.player and total > 0
        ]
        self._enemy_reach_cache = {}

    def _total_ships(self, owner):
        return sum(p.ships for p in self.planets if p.owner == owner) + sum(
            f.ships for f in self.fleets if f.owner == owner
        )

    def _build_planet_path_cache(self, horizon):
        cache = {}
        angular_velocity = obs_get(self.obs, "angular_velocity", 0.0)
        for planet in self.planets:
            paths = [None] * (horizon + 1)
            if planet.id in self.comet_ids:
                data = _comet_path_data(planet.id, self.obs)
                if data is None:
                    cache[planet.id] = paths
                    continue
                group, comet_path = data
                current_idx = int(group.get("path_index", -1))
                for offset in range(1, horizon + 1):
                    old_idx = current_idx + offset - 1
                    new_idx = current_idx + offset
                    if old_idx >= len(comet_path):
                        break
                    if old_idx < 0:
                        old_pos = (planet.x, planet.y)
                        check = old_pos[0] >= 0.0 and old_pos[1] >= 0.0
                    else:
                        old_pos = (
                            float(comet_path[old_idx][0]),
                            float(comet_path[old_idx][1]),
                        )
                        check = True
                    if new_idx >= len(comet_path):
                        new_pos = old_pos
                    elif new_idx < 0:
                        new_pos = (planet.x, planet.y)
                    else:
                        new_pos = (
                            float(comet_path[new_idx][0]),
                            float(comet_path[new_idx][1]),
                        )
                    paths[offset] = (old_pos, new_pos, check)
                cache[planet.id] = paths
                continue

            initial = self.initial_by_id.get(planet.id)
            if initial is None:
                for offset in range(1, horizon + 1):
                    pos = (planet.x, planet.y)
                    paths[offset] = (pos, pos, True)
                cache[planet.id] = paths
                continue

            orbital_radius = dist((initial.x, initial.y), (CENTER, CENTER))
            if orbital_radius + planet.radius >= ROTATION_RADIUS_LIMIT:
                for offset in range(1, horizon + 1):
                    pos = (planet.x, planet.y)
                    paths[offset] = (pos, pos, True)
                cache[planet.id] = paths
                continue

            angle0 = math.atan2(initial.y - CENTER, initial.x - CENTER)
            positions = []
            for offset in range(0, horizon + 1):
                step = self.step + offset
                angle = angle0 + angular_velocity * _position_step_index(step)
                positions.append(
                    (
                        CENTER + orbital_radius * math.cos(angle),
                        CENTER + orbital_radius * math.sin(angle),
                    )
                )
            for offset in range(1, horizon + 1):
                paths[offset] = (positions[offset - 1], positions[offset], True)
            cache[planet.id] = paths
        return cache

    def turns_left(self):
        return max(0, 499 - self.step)

    def reserve_for(self, planet):
        if planet.id in self.comet_ids:
            return 0

        loss_turn = self.ledger.first_not_owned_turn(
            planet.id, self.player, DEFENSE_HORIZON
        )
        if loss_turn is not None and loss_turn <= 8:
            return 0

        if self.step > ENDGAME_STEP:
            return 0 if self.my_total <= self.enemy_total * 1.2 else max(1, planet.production)

        if self.step < 35:
            reserve = max(2, int(planet.production * 1.2))
        elif self.step < 120:
            reserve = max(4, int(planet.production * 2.0))
        else:
            reserve = max(5, int(planet.production * 2.7))

        if self.my_total > self.enemy_total * 1.7 and self.step > 120:
            reserve = max(2, reserve // 2)
        if self.my_production < self.enemy_production and self.step > 150:
            reserve = max(1, reserve - planet.production)
        return reserve

    def initial_available(self):
        return {
            planet.id: max(0, int(planet.ships - self.reserve_for(planet)))
            for planet in self.my_planets
        }

    def enemy_available_for(self, planet):
        if planet.id in self.comet_ids:
            return max(0, int(planet.ships))
        reserve = max(1, int(planet.production * 1.7))
        if self.step > ENDGAME_STEP:
            reserve = 0
        return max(0, int(planet.ships - reserve))

    def enemy_reach(self, target):
        if target.id in self._enemy_reach_cache:
            return self._enemy_reach_cache[target.id]

        best = (999.0, 0, None)
        horizon = min(MAX_ETA, max(1, self.turns_left() - 1))
        for source in self.enemy_planets:
            if source.id == target.id:
                continue
            available = self.enemy_available_for(source)
            if available <= 0:
                continue
            intercept = solve_intercept_angle(
                (source.x, source.y), target.id, available, self.step, self.obs
            )
            if intercept is None:
                continue
            _, eta = intercept
            if eta > horizon:
                continue
            if eta < best[0]:
                best = (eta, available, source.id)

        self._enemy_reach_cache[target.id] = best
        return best

    def race_buffer(self, target, our_eta, needed):
        enemy_eta, enemy_ships, _ = self.enemy_reach(target)
        if enemy_eta >= 998.0:
            return 0
        margin = enemy_eta - our_eta
        if margin < -2.0:
            return 10**6
        if margin <= 4.0:
            return max(3, int(enemy_ships * 0.20) + target.production)
        if margin <= 10.0 and enemy_ships >= needed:
            return max(2, int(enemy_ships * 0.12))
        return 0


def _turn_weight(state, eta):
    horizon = 85 if state.step < 130 else 120
    if state.step > ENDGAME_STEP:
        horizon = state.turns_left()
    return max(0.0, min(float(horizon), float(state.turns_left()) - eta))


def _nearest_owned_distance(state, target):
    if not state.my_planets:
        return 999.0
    return min(dist((p.x, p.y), (target.x, target.y)) for p in state.my_planets)


def _send_amount_with_speed_bonus(needed, available, distance_to_target, state):
    needed = int(max(1, needed))
    if available < needed:
        return 0
    send = needed
    if distance_to_target > 34:
        send = max(send, int(needed * 1.16) + 2)
    if distance_to_target > 52:
        send = max(send, int(needed * 1.35) + 4)
    if distance_to_target > 70:
        send = max(send, int(needed * 1.58) + 6)
    if state.step < 55 and needed <= 13 and available >= needed + 5:
        send = max(send, needed + 5)
    if state.step > ENDGAME_STEP:
        send = needed
    return min(available, send)


def _validated_intercept(source, target, ships, state, horizon=MAX_ETA):
    intercept = solve_intercept_angle(
        (source.x, source.y), target.id, ships, state.step, state.obs
    )
    if intercept is None:
        return None
    angle, eta = intercept
    max_horizon = min(
        int(horizon),
        int(math.ceil(eta)) + 2,
        max(1, state.turns_left() - 1),
    )
    hit_id, hit_turn = first_collision_for_launch(source, angle, ships, state, max_horizon)
    if hit_id == target.id:
        return angle, float(hit_turn)

    # Small offsets often turn a center-line miss or obstruction into a valid
    # edge hit on the target disk while preserving the same high-level intent.
    for degrees in (1.5, -1.5, 4.0, -4.0):
        test_angle = angle + math.radians(degrees)
        hit_id, hit_turn = first_collision_for_launch(
            source, test_angle, ships, state, max_horizon
        )
        if hit_id == target.id:
            return test_angle, float(hit_turn)
    return None


def _ships_for_speed(speed):
    speed = max(1.0, min(MAX_FLEET_SPEED, float(speed)))
    if speed <= 1.0:
        return 1
    ratio = ((speed - 1.0) / (MAX_FLEET_SPEED - 1.0)) ** (2.0 / 3.0)
    return max(1, int(math.ceil(math.exp(ratio * math.log(1000)))))


def _aligned_part_for_turn(source, target, desired_turn, max_send, state):
    desired_turn = int(max(1, desired_turn))
    target_xy = _target_position(
        target.id, state.step, state.step + desired_turn, state.obs
    )
    if target_xy is None:
        return None
    distance_to_target = dist((source.x, source.y), target_xy)
    base = _ships_for_speed(distance_to_target / max(1.0, desired_turn))
    candidates = {
        max(1, min(max_send, base)),
        max(1, min(max_send, base + 2)),
        max(1, min(max_send, int(base * 1.15) + 2)),
        max(1, min(max_send, int(base * 1.35) + 4)),
        max(1, min(max_send, int(base * 0.85))),
        max_send,
    }
    best = None
    for ships in sorted(candidates, reverse=True):
        if ships <= 0 or ships > max_send:
            continue
        intercept = _validated_intercept(source, target, ships, state, desired_turn + 2)
        if intercept is None:
            continue
        angle, eta = intercept
        if abs(eta - desired_turn) <= 1.0:
            part = (source.id, angle, ships, eta)
            if best is None or part[2] > best[2]:
                best = part
    return best


def _candidate_score(state, target, send, eta, kind):
    turns = _turn_weight(state, eta)
    owner, garrison = state.ledger.state_at(target.id, eta)
    if owner == state.player:
        return -9999.0

    prod_gain = target.production * turns
    enemy_multiplier = 1.0
    if owner != -1:
        enemy_multiplier = 2.1
    if kind == "comet":
        enemy_multiplier = 0.55
        prod_gain = min(prod_gain, 28.0)

    score = prod_gain * enemy_multiplier
    score += target.production * 13.0
    score -= send * 0.74
    score -= eta * 0.42
    score -= max(0.0, garrison - target.ships) * 0.12

    enemy_eta, enemy_ships, _ = state.enemy_reach(target)
    race_margin = enemy_eta - eta
    if owner == -1:
        if race_margin < -1.0:
            score -= 90.0 + target.production * 14.0
        elif race_margin <= 3.0:
            score -= 35.0 + enemy_ships * 0.18
        elif race_margin >= 9.0 and state.step < 180:
            score += target.production * 6.0
    elif owner not in (-1, state.player):
        if race_margin <= 4.0:
            score -= min(70.0, enemy_ships * 0.35 + 20.0)

    if target.production >= 4:
        score += 18.0
    if state.step < 70 and owner == -1:
        score += target.production * 7.0
        score -= target.ships * 0.18
    if state.step < OPENING_END_STEP and owner == -1:
        score += max(0.0, target.production * 16.0 - target.ships * 0.32 - eta * 0.55)
    if owner not in (-1, state.player):
        score += target.ships * 0.25
        if state.my_production < state.enemy_production:
            score += target.production * 9.0
    if state.step > ENDGAME_STEP:
        remaining = max(0.0, state.turns_left() - eta)
        if owner == -1:
            score = target.production * remaining - target.ships * 1.3 - send * 0.22
        else:
            score = target.ships * 1.85 + target.production * remaining * 1.7
            score -= send * 0.48 + eta * 0.14
    if target.id in state.comet_ids and eta > 16:
        score -= 80.0
    return score


def _build_single_capture_candidate(state, target, source, available, kind):
    source_available = available.get(source.id, 0)
    if source_available <= 0:
        return None

    source_xy = (source.x, source.y)
    probe = max(min(source_available, target.ships + 35), target.ships + CAPTURE_BUFFER)
    probe = max(1, min(source_available, probe))
    intercept = _validated_intercept(source, target, probe, state)
    if intercept is None:
        return None

    angle, eta = intercept
    if eta > state.turns_left() - 1:
        return None
    if kind == "comet" and eta > 18:
        return None

    needed = state.ledger.needed_to_capture(target.id, eta, state.player)
    race_buffer = state.race_buffer(target, eta, needed)
    if race_buffer >= 10**6:
        return None
    needed += race_buffer
    target_xy = _target_position(target.id, state.step, state.step + eta, state.obs)
    if target_xy is None:
        return None
    distance_to_target = dist(source_xy, target_xy)
    send = _send_amount_with_speed_bonus(needed, source_available, distance_to_target, state)
    if send < needed:
        return None

    intercept = _validated_intercept(source, target, send, state)
    if intercept is None:
        return None
    angle, eta = intercept
    if eta > state.turns_left() - 1:
        return None

    needed = state.ledger.needed_to_capture(target.id, eta, state.player)
    race_buffer = state.race_buffer(target, eta, needed)
    if race_buffer >= 10**6:
        return None
    needed += race_buffer
    send = _send_amount_with_speed_bonus(needed, source_available, distance_to_target, state)
    if send < needed:
        return None

    intercept = _validated_intercept(source, target, send, state)
    if intercept is None:
        return None
    angle, eta = intercept

    score = _candidate_score(state, target, send, eta, kind)
    return Candidate(kind, score, target.id, ((source.id, angle, send),), eta, send, kind)


def _build_multi_attack_candidate(state, target, available):
    options = []
    sources = sorted(
        [p for p in state.my_planets if available.get(p.id, 0) > 0],
        key=lambda p: (
            dist((p.x, p.y), (target.x, target.y)),
            -available.get(p.id, 0),
        ),
    )[:5]
    for source in sources:
        if available.get(source.id, 0) <= 0:
            continue
        probe = available[source.id]
        intercept = _validated_intercept(source, target, probe, state)
        if intercept is None:
            continue
        angle, eta = intercept
        if eta > min(80, state.turns_left() - 1):
            continue
        options.append((eta, source, angle, available[source.id]))

    if len(options) < 2:
        return None
    options.sort(key=lambda item: item[0])

    best = None
    anchor_turns = sorted({int(math.ceil(item[0])) for item in options})[:3]
    for target_turn in anchor_turns:
        aligned = []
        for _, source, _, source_available in options:
            if source_available <= 0:
                continue
            part = _aligned_part_for_turn(
                source, target, target_turn, source_available, state
            )
            if part is not None:
                aligned.append((source, part))
        if len(aligned) < 2:
            continue

        needed = state.ledger.needed_to_capture(target.id, target_turn, state.player) + 4
        race_buffer = state.race_buffer(target, float(target_turn), needed)
        if race_buffer >= 10**6:
            continue
        needed += race_buffer
        parts = []
        committed = 0
        for source, part in sorted(aligned, key=lambda item: item[1][2], reverse=True):
            remaining = needed - committed
            if remaining <= 0:
                break
            source_id, angle, max_aligned_send, _ = part
            send = min(max_aligned_send, remaining)
            if send <= 0 or available.get(source.id, 0) < send:
                continue
            if send != max_aligned_send:
                trimmed = _aligned_part_for_turn(source, target, target_turn, send, state)
                if trimmed is None:
                    continue
                source_id, angle, send, _ = trimmed
            parts.append((source_id, angle, send))
            committed += send

        if committed < needed:
            continue
        score = _candidate_score(state, target, committed, float(target_turn), "attack")
        score += target.production * 12.0
        candidate = Candidate(
            "attack",
            score,
            target.id,
            tuple(parts),
            float(target_turn),
            committed,
            "coordinated",
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _target_priority(state, target):
    owner, garrison = state.ledger.state_at(target.id, 1)
    priority = target.production * 12.0 - garrison * 0.2
    priority -= _nearest_owned_distance(state, target) * 0.12
    enemy_eta, _, _ = state.enemy_reach(target)
    if owner == -1 and enemy_eta < 18:
        priority += target.production * 3.5 - max(0.0, 18.0 - enemy_eta)
    if owner not in (-1, state.player):
        priority += target.production * 8.0 + target.ships * 0.15
    if target.id in state.comet_ids:
        priority = 8.0 - target.ships * 0.4 - _nearest_owned_distance(state, target) * 0.1
    if state.step > ENDGAME_STEP and owner != -1:
        priority += target.ships * 0.5
    return priority


def _candidate_targets(state, claimed_targets):
    targets = [
        p
        for p in state.planets
        if p.owner != state.player and p.id not in claimed_targets
    ]
    filtered = []
    for target in targets:
        if target.id in state.comet_ids:
            if target.owner == state.player or target.ships > 16:
                continue
        filtered.append(target)
    filtered.sort(key=lambda p: _target_priority(state, p), reverse=True)
    return filtered[:14]


def _generate_capture_candidates(state, available, claimed_targets):
    candidates = []
    for target in _candidate_targets(state, claimed_targets):
        owner, _ = state.ledger.state_at(target.id, 1)
        if target.id in state.comet_ids:
            kind = "comet"
        elif owner == -1:
            kind = "expand"
        else:
            kind = "attack"

        sources = sorted(
            [p for p in state.my_planets if available.get(p.id, 0) > 0],
            key=lambda p: (
                dist((p.x, p.y), (target.x, target.y)),
                -available.get(p.id, 0),
            ),
        )[:5]
        for source in sources:
            candidate = _build_single_capture_candidate(
                state, target, source, available, kind
            )
            if candidate is not None:
                candidates.append(candidate)

        if kind == "attack" and target.production >= 4 and state.step > OPENING_END_STEP:
            candidate = _build_multi_attack_candidate(state, target, available)
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _generate_defense_candidates(state, available):
    candidates = []
    for target in state.my_planets:
        loss_turn = state.ledger.first_not_owned_turn(
            target.id, state.player, DEFENSE_HORIZON
        )
        if loss_turn is None:
            continue

        lost_owner, lost_garrison = state.ledger.state_at(target.id, loss_turn)
        needed = max(1, int(math.ceil(lost_garrison + 5)))
        parts = []
        committed = 0
        sources = sorted(
            [p for p in state.my_planets if p.id != target.id and available.get(p.id, 0) > 0],
            key=lambda p: dist((p.x, p.y), (target.x, target.y)),
        )
        for source in sources:
            if committed >= needed:
                break
            send = min(available[source.id], needed - committed)
            intercept = _validated_intercept(source, target, send, state, loss_turn + 1)
            if intercept is None:
                continue
            angle, eta = intercept
            if eta > loss_turn + 0.5:
                continue
            parts.append((source.id, angle, send))
            committed += send

        if committed >= needed:
            score = 1000.0 + target.production * 35.0 + target.ships - loss_turn * 8.0
            if lost_owner not in (-1, state.player):
                score += 80.0
            candidates.append(
                Candidate(
                    "defend",
                    score,
                    target.id,
                    tuple(parts),
                    float(loss_turn),
                    committed,
                    "save_planet",
                )
            )
    return candidates


def _generate_evacuation_candidates(state, available, protected_targets):
    candidates = []
    safe_destinations = [
        planet
        for planet in state.my_planets
        if state.ledger.first_not_owned_turn(planet.id, state.player, DEFENSE_HORIZON)
        is None
    ]
    if not safe_destinations:
        return candidates

    for source in state.my_planets:
        if source.id in protected_targets:
            continue
        loss_turn = state.ledger.first_not_owned_turn(
            source.id, state.player, EVACUATE_HORIZON
        )
        if loss_turn is None:
            continue
        send = available.get(source.id, 0)
        if send <= max(2, source.production):
            continue

        destinations = sorted(
            [p for p in safe_destinations if p.id != source.id],
            key=lambda p: (
                -p.production,
                dist((source.x, source.y), (p.x, p.y)),
                -p.ships,
            ),
        )
        for dest in destinations[:5]:
            intercept = _validated_intercept(
                source, dest, send, state, min(MAX_ETA, state.turns_left() - 1)
            )
            if intercept is None:
                continue
            angle, eta = intercept
            score = 650.0 + send * 1.2 + source.production * 24.0 - loss_turn * 13.0
            candidates.append(
                Candidate(
                    "evacuate",
                    score,
                    dest.id,
                    ((source.id, angle, send),),
                    eta,
                    send,
                    "evacuate_doomed",
                )
            )
            break
    return candidates


def _apply_candidate(candidate, available, moves):
    if len(moves) + len(candidate.parts) > MAX_MOVES:
        return False
    for source_id, _, ships in candidate.parts:
        if available.get(source_id, 0) < ships or ships <= 0:
            return False
    for source_id, angle, ships in candidate.parts:
        moves.append([int(source_id), float(angle), int(ships)])
        available[source_id] -= int(ships)
    return True


def _selection_threshold(state, candidate):
    if candidate.kind in ("defend", "evacuate"):
        return -9999.0
    if state.step > ENDGAME_STEP:
        return -8.0
    if state.step < 70 and candidate.kind == "expand":
        return 6.0
    if candidate.kind == "attack":
        return 18.0
    if candidate.kind == "comet":
        return 8.0
    return 10.0


def _choose_moves(state):
    available = state.initial_available()
    moves = []
    claimed_targets = set()

    defense_candidates = sorted(
        _generate_defense_candidates(state, available),
        key=lambda c: c.score,
        reverse=True,
    )
    protected_targets = set()
    for candidate in defense_candidates:
        if _apply_candidate(candidate, available, moves):
            claimed_targets.add(candidate.target_id)
            protected_targets.add(candidate.target_id)
        if len(moves) >= MAX_MOVES:
            return moves[:MAX_MOVES]

    evacuation_candidates = sorted(
        _generate_evacuation_candidates(state, available, protected_targets),
        key=lambda c: c.score,
        reverse=True,
    )
    for candidate in evacuation_candidates:
        if _apply_candidate(candidate, available, moves):
            claimed_targets.add(candidate.target_id)
        if len(moves) >= MAX_MOVES:
            return moves[:MAX_MOVES]

    for _ in range(8):
        candidates = _generate_capture_candidates(state, available, claimed_targets)
        if not candidates:
            break
        candidates.sort(key=lambda c: c.score, reverse=True)
        chosen = candidates[0]
        if chosen.score < _selection_threshold(state, chosen):
            break
        if _apply_candidate(chosen, available, moves):
            claimed_targets.add(chosen.target_id)
        else:
            claimed_targets.add(chosen.target_id)
        if len(moves) >= MAX_MOVES:
            break

    return moves[:MAX_MOVES]


def ships_needed_to_capture(target_planet, eta, current_step, obs):
    if (
        _CURRENT_STATE is not None
        and _CURRENT_STATE.obs is obs
        and target_planet.id in _CURRENT_STATE.planet_by_id
    ):
        return _CURRENT_STATE.ledger.needed_to_capture(
            target_planet.id, eta, _CURRENT_STATE.player
        )

    player = obs_get(obs, "player", 0)
    eta = max(0.0, float(eta))
    needed = float(target_planet.ships)
    if target_planet.owner != -1:
        needed += target_planet.production * eta

    horizon = min(MAX_ETA, int(math.ceil(eta)) + 2)
    for fleet in as_fleets(obs):
        arrival = _fleet_arrival_eta(fleet, target_planet, current_step, obs, horizon)
        if arrival is None or arrival > eta + 1.0:
            continue
        if fleet.owner == player:
            needed -= fleet.ships * 0.9
        elif target_planet.owner != -1 and fleet.owner == target_planet.owner:
            needed += fleet.ships
        else:
            needed += fleet.ships * 0.35

    return max(1, int(math.ceil(needed + CAPTURE_BUFFER)))


def agent(obs, config=None):
    del config
    global _CURRENT_STATE
    state = GameState(obs)
    _CURRENT_STATE = state
    if not state.my_planets:
        return []
    return _choose_moves(state)


if __name__ == "__main__":
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["main.py", "random"])
    final = env.steps[-1]
    for i, state in enumerate(final):
        print(f"Player {i}: reward={state.reward}, status={state.status}")
