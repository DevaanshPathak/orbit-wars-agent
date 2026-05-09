import math
import time
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

OPENING_PLANNER_LIMIT = 50
OPENING_PLANNER_BUDGET = 0.18
OPENING_PLAN_DEPTH_ONE = 4
OPENING_PLAN_DEPTH_FEW = 3
OPENING_PLAN_DEPTH_MANY = 2
OPENING_PLAN_BEAM = 5
OPENING_PLAN_TARGETS = 10
OPENING_PLAN_WAIT = 12
OPENING_PLAN_MAX_MY_PLANETS = 5
OPENING_PLAN_DEFENSE_GATE = 15

SNIPE_HORIZON = 55
RECAPTURE_WINDOW = 10
STAGING_MIN_SHIPS = 12
STAGING_MAX_ETA = 42
PARTIAL_SOURCE_MIN_SHIPS = 6
PROACTIVE_DEFENSE_HORIZON = 12

USE_OPENING_PLANNER = True
USE_SNIPES = True
USE_RECAPTURE = True
USE_STAGING = False
USE_CRASH_EXPLOITS = True
USE_ENDGAME_SCORE_MODE = True
USE_MODEL_SCORER = False
USE_DEEP_PLANNER = True
SPECULATIVE_TIME_MARGIN = 0.14
MODEL_BLEND = 0.22
PLANNER_HORIZON = 32
PLANNER_BEAM = 3
PLANNER_TOP_CANDIDATES = 8
PLANNER_MAX_PICKS = 3
PLANNER_BUDGET = 0.055

# Trained model artifacts must not be committed to GitHub. A local, gitignored
# submission build may replace this with the JSON exported by the v5 notebook.
MODEL_WEIGHTS = None

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


def planet_is_static(planet, initial_by_id=None):
    initial = initial_by_id.get(planet.id) if initial_by_id is not None else None
    base = initial if initial is not None else planet
    return dist((base.x, base.y), (CENTER, CENTER)) + planet.radius >= ROTATION_RADIUS_LIMIT


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
        self.static_neutral_planets = [
            p for p in self.neutral_planets if planet_is_static(p, self.initial_by_id)
        ]
        self.comet_planets = [p for p in self.planets if p.id in self.comet_ids]
        self.planet_path_cache = self._build_planet_path_cache(
            max(LEDGER_HORIZON, MAX_ETA) + 2
        )
        self.ledger = ArrivalLedger(self)
        self.owner_strength = {owner: self._total_ships(owner) for owner in range(4)}
        self.owner_production = {
            owner: sum(p.production for p in self.planets if p.owner == owner)
            for owner in range(4)
        }
        self.totals = self.owner_strength
        self.my_total = self.totals.get(self.player, 0)
        self.enemy_total = sum(
            total for owner, total in self.totals.items() if owner != self.player
        )
        self.my_production = sum(p.production for p in self.my_planets)
        self.enemy_production = sum(p.production for p in self.enemy_planets)
        self.max_enemy_total = max(
            (total for owner, total in self.totals.items() if owner != self.player),
            default=0,
        )
        self.enemy_players = [
            owner for owner, total in self.totals.items() if owner != self.player and total > 0
        ]
        self.num_players = 1 + len(self.enemy_players)
        self._enemy_reach_cache = {}
        self._my_reach_cache = {}
        self._projected_state_cache = {}

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

    def is_static(self, planet):
        return planet_is_static(planet, self.initial_by_id)

    def my_reach(self, target):
        if target.id in self._my_reach_cache:
            return self._my_reach_cache[target.id]

        best = (999.0, 0, None)
        horizon = min(MAX_ETA, max(1, self.turns_left() - 1))
        for source in self.my_planets:
            if source.id == target.id:
                continue
            available = max(0, int(source.ships - self.reserve_for(source)))
            if available <= 0:
                continue
            intercept = solve_intercept_angle(
                (source.x, source.y), target.id, available, self.step, self.obs
            )
            if intercept is None:
                continue
            _, eta = intercept
            if eta <= horizon and eta < best[0]:
                best = (eta, available, source.id)

        self._my_reach_cache[target.id] = best
        return best

    def reaction_times(self, target):
        return self.my_reach(target)[0], self.enemy_reach(target)[0]

    def projected_state(
        self,
        planet_id,
        eval_turn,
        planned_commitments=None,
        extra_arrivals=(),
    ):
        eval_turn = max(0, int(math.ceil(eval_turn)))
        planned_commitments = planned_commitments or {}
        normalized_extra = tuple(
            (max(1, int(math.ceil(turn))), int(owner), int(ships))
            for turn, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turn))) <= eval_turn
        )
        cache_key = None
        if not planned_commitments.get(planet_id) and not normalized_extra:
            cache_key = (planet_id, eval_turn)
            cached = self._projected_state_cache.get(cache_key)
            if cached is not None:
                return cached

        planet = self.planet_by_id.get(planet_id)
        if planet is None:
            return -1, 0.0

        by_turn = {}
        for turn, arrivals in self.ledger.arrivals.get(planet_id, {}).items():
            turn = max(1, int(math.ceil(turn)))
            if turn > eval_turn:
                continue
            slot = by_turn.setdefault(turn, {})
            for owner, ships in arrivals.items():
                slot[owner] = slot.get(owner, 0) + int(ships)

        for turn, owner, ships in planned_commitments.get(planet_id, []):
            turn = max(1, int(math.ceil(turn)))
            if turn > eval_turn or ships <= 0:
                continue
            slot = by_turn.setdefault(turn, {})
            slot[owner] = slot.get(owner, 0) + int(ships)

        for turn, owner, ships in normalized_extra:
            slot = by_turn.setdefault(turn, {})
            slot[owner] = slot.get(owner, 0) + int(ships)

        owner = planet.owner
        garrison = float(planet.ships)
        for turn in range(1, eval_turn + 1):
            if owner != -1:
                garrison += planet.production
            owner, garrison = _resolve_combat(owner, garrison, by_turn.get(turn, {}))

        result = (owner, max(0.0, garrison))
        if cache_key is not None:
            self._projected_state_cache[cache_key] = result
        return result

    def _ownership_search_cap(self, eval_turn):
        visible = sum(int(p.ships) for p in self.planets) + sum(
            int(f.ships) for f in self.fleets
        )
        production = sum(int(p.production) for p in self.planets)
        return max(32, int(visible + production * max(2, eval_turn + 2) + 32))

    def min_ships_to_own_by(
        self,
        planet_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        upper_bound=None,
    ):
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            return (int(upper_bound) + 1) if upper_bound is not None else self._ownership_search_cap(eval_turn) + 1

        owner_before, ships_before = self.projected_state(
            planet_id, eval_turn, planned_commitments=planned_commitments
        )
        if owner_before == attacker_owner:
            return 0

        def owns_with(ships):
            owner_after, _ = self.projected_state(
                planet_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, attacker_owner, int(ships)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_with(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships_before)) + 1)
            cap = self._ownership_search_cap(eval_turn)
            while hi < cap and not owns_with(hi):
                hi *= 2
            hi = min(hi, cap)
            if not owns_with(hi):
                return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_with(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def min_ships_to_own_at(
        self,
        planet_id,
        arrival_turn,
        attacker_owner=None,
        planned_commitments=None,
        upper_bound=None,
    ):
        attacker = self.player if attacker_owner is None else attacker_owner
        return self.min_ships_to_own_by(
            planet_id,
            arrival_turn,
            attacker,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        def holds_with(ships):
            for turn in range(arrival_turn, hold_until + 1):
                owner, _ = self.projected_state(
                    planet_id,
                    turn,
                    planned_commitments=planned_commitments,
                    extra_arrivals=((arrival_turn, self.player, int(ships)),),
                )
                if owner != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with(hi):
                return hi + 1
        else:
            hi = 1
            cap = self._ownership_search_cap(hold_until)
            while hi < cap and not holds_with(hi):
                hi *= 2
            hi = min(hi, cap)
            if not holds_with(hi):
                return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

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


def _nearest_distance_to_set(point, planets):
    if not planets:
        return 999.0
    return min(dist(point, (planet.x, planet.y)) for planet in planets)


def _build_policy(state):
    reaction_time_map = {}
    indirect_value = {}
    attack_budget = {}
    reserve = {}

    for target in state.planets:
        if target.owner != state.player:
            reaction_time_map[target.id] = state.reaction_times(target)

        nearby_neutral = 0.0
        nearby_enemy = 0.0
        nearby_mine = 0.0
        for other in state.planets:
            if other.id == target.id:
                continue
            weight = other.production / max(8.0, dist((target.x, target.y), (other.x, other.y)))
            if other.owner == state.player:
                nearby_mine += weight
            elif other.owner == -1:
                nearby_neutral += weight
            else:
                nearby_enemy += weight
        indirect_value[target.id] = nearby_neutral * 0.9 + nearby_enemy * 1.25 + nearby_mine * 0.35

    for planet in state.my_planets:
        keep = state.reserve_for(planet)
        enemy_eta, enemy_ships, _ = state.enemy_reach(planet)
        if enemy_eta <= PROACTIVE_DEFENSE_HORIZON:
            keep = max(keep, int(enemy_ships * 0.18) + planet.production)
        reserve[planet.id] = min(int(planet.ships), max(0, int(keep)))
        attack_budget[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

    return {
        "reaction_time_map": reaction_time_map,
        "indirect_value": indirect_value,
        "reserve": reserve,
        "attack_budget": attack_budget,
    }


def _policy_reaction_times(policy, target_id):
    return policy["reaction_time_map"].get(target_id, (999.0, 999.0))


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
    for ships in sorted(candidates):
        if ships <= 0 or ships > max_send:
            continue
        intercept = _validated_intercept(source, target, ships, state, desired_turn + 2)
        if intercept is None:
            continue
        angle, eta = intercept
        if abs(eta - desired_turn) <= 0.75:
            part = (source.id, angle, ships, eta)
            if best is None or part[2] < best[2]:
                best = part
    return best


def _candidate_score(state, target, send, eta, kind, policy=None):
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
    if policy is not None:
        score += policy["indirect_value"].get(target.id, 0.0) * turns * 0.08

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
    if state.is_static(target):
        score += 12.0
        if state.step < 90 and owner == -1:
            score += target.production * 8.0
    elif state.step < OPENING_END_STEP and owner == -1:
        my_t, enemy_t = _policy_reaction_times(policy, target.id) if policy else state.reaction_times(target)
        if eta > 13.0 and enemy_t - my_t < 3.0:
            score -= 24.0
        if target.production <= 2:
            score -= 12.0
    if state.step < 70 and owner == -1:
        score += target.production * 7.0
        score -= target.ships * 0.18
    if state.step < OPENING_END_STEP and owner == -1:
        score += max(0.0, target.production * 16.0 - target.ships * 0.32 - eta * 0.55)
    if owner not in (-1, state.player):
        score += target.ships * 0.25
        if state.my_production < state.enemy_production:
            score += target.production * 9.0
    if USE_ENDGAME_SCORE_MODE and state.step > ENDGAME_STEP:
        remaining = max(0.0, state.turns_left() - eta)
        if owner == -1:
            score = target.production * remaining - target.ships * 1.3 - send * 0.22
        else:
            score = target.ships * 1.85 + target.production * remaining * 1.7
            score -= send * 0.48 + eta * 0.14
    if target.id in state.comet_ids and eta > 16:
        score -= 80.0
    return score


def _candidate_features(state, candidate, policy=None):
    target = state.planet_by_id.get(candidate.target_id)
    if target is None:
        return {}

    eta = max(1.0, float(candidate.eta))
    owner_at_eta, garrison_at_eta = state.ledger.state_at(target.id, eta)
    enemy_eta, enemy_ships, _ = state.enemy_reach(target)
    my_eta, my_ships, _ = state.my_reach(target)
    enemy_eta_capped = min(999.0, float(enemy_eta))
    my_eta_capped = min(999.0, float(my_eta))
    source_distances = []
    source_reserve_sum = 0.0
    source_budget_sum = 0.0
    min_source_reserve = 999.0
    if policy is None:
        policy = {}
    reserve = policy.get("reserve", {})
    attack_budget = policy.get("attack_budget", {})
    indirect_value = policy.get("indirect_value", {})

    for source_id, _, _ in candidate.parts:
        source = state.planet_by_id.get(source_id)
        if source is None:
            continue
        d = dist((source.x, source.y), (target.x, target.y))
        source_distances.append(d)
        source_reserve = float(reserve.get(source_id, state.reserve_for(source)))
        min_source_reserve = min(min_source_reserve, source_reserve)
        source_reserve_sum += source_reserve
        source_budget_sum += float(attack_budget.get(source_id, 0))

    if not source_distances:
        source_distances = [999.0]
        min_source_reserve = 0.0

    total_visible = max(1.0, float(state.my_total + state.enemy_total))
    ships_sent = float(candidate.ships)
    race_margin = enemy_eta_capped - eta
    kind = candidate.kind

    features = {
        "step": float(state.step),
        "turns_left": float(state.turns_left()),
        "num_players": float(state.num_players),
        "my_planets": float(len(state.my_planets)),
        "enemy_planets": float(len(state.enemy_planets)),
        "neutral_planets": float(len(state.neutral_planets)),
        "my_total": float(state.my_total),
        "enemy_total": float(state.enemy_total),
        "max_enemy_total": float(state.max_enemy_total),
        "my_production": float(state.my_production),
        "enemy_production": float(state.enemy_production),
        "production_gap": float(state.my_production - state.enemy_production),
        "target_owner_neutral": 1.0 if target.owner == -1 else 0.0,
        "target_owner_enemy": 1.0 if target.owner not in (-1, state.player) else 0.0,
        "target_owner_projected_mine": 1.0 if owner_at_eta == state.player else 0.0,
        "target_ships": float(target.ships),
        "target_projected_garrison": float(garrison_at_eta),
        "target_production": float(target.production),
        "target_static": 1.0 if state.is_static(target) else 0.0,
        "target_orbiting": 0.0 if state.is_static(target) else 1.0,
        "target_comet": 1.0 if target.id in state.comet_ids else 0.0,
        "eta": eta,
        "ships_sent": ships_sent,
        "parts_count": float(len(candidate.parts)),
        "ships_per_eta": ships_sent / eta,
        "ship_cost_fraction": ships_sent / total_visible,
        "source_distance_min": float(min(source_distances)),
        "source_distance_avg": float(sum(source_distances) / len(source_distances)),
        "source_reserve_min": float(min_source_reserve),
        "source_reserve_sum": float(source_reserve_sum),
        "source_budget_sum": float(source_budget_sum),
        "enemy_eta": enemy_eta_capped,
        "enemy_ships": float(enemy_ships),
        "my_eta": my_eta_capped,
        "my_reach_ships": float(my_ships),
        "race_margin": float(race_margin),
        "indirect_value": float(indirect_value.get(target.id, 0.0)),
        "heuristic_score_scaled": float(candidate.score) / 100.0,
        "kind_expand": 1.0 if kind == "expand" else 0.0,
        "kind_attack": 1.0 if kind == "attack" else 0.0,
        "kind_comet": 1.0 if kind == "comet" else 0.0,
        "kind_snipe": 1.0 if kind == "snipe" else 0.0,
        "kind_recapture": 1.0 if kind == "recapture" else 0.0,
        "kind_crash": 1.0 if kind == "crash" else 0.0,
        "kind_stage": 1.0 if kind == "stage" else 0.0,
        "kind_defend": 1.0 if kind == "defend" else 0.0,
        "kind_evacuate": 1.0 if kind == "evacuate" else 0.0,
    }
    return features


def _model_score_candidate(features, model=None):
    model = MODEL_WEIGHTS if model is None else model
    if not model:
        return 0.0
    weights = model.get("weights", {})
    if not weights:
        return 0.0
    means = model.get("mean", {})
    scales = model.get("scale", {})
    score = float(model.get("bias", 0.0))
    for name, weight in weights.items():
        value = float(features.get(name, 0.0))
        scale = float(scales.get(name, 1.0) or 1.0)
        value = (value - float(means.get(name, 0.0))) / scale
        score += float(weight) * value

    model_type = str(model.get("model_type", ""))
    if model_type.startswith("logistic"):
        if score >= 0:
            prob = 1.0 / (1.0 + math.exp(-min(50.0, score)))
        else:
            exp_score = math.exp(max(-50.0, score))
            prob = exp_score / (1.0 + exp_score)
        return (prob - 0.5) * 120.0
    return score


def _score_candidate_v5(state, candidate, policy=None):
    score = float(candidate.score)
    if USE_MODEL_SCORER and MODEL_WEIGHTS:
        features = _candidate_features(state, candidate, policy)
        model_score = _model_score_candidate(features, MODEL_WEIGHTS)
        score = score * (1.0 - MODEL_BLEND) + (score + model_score) * MODEL_BLEND
    return score


def _planner_projected_value(state, candidate, policy=None):
    target = state.planet_by_id.get(candidate.target_id)
    if target is None:
        return -9999.0
    remaining = max(0.0, min(float(PLANNER_HORIZON), state.turns_left() - candidate.eta))
    if remaining <= 0:
        return -candidate.ships * 0.12

    owner_at_eta, garrison_at_eta = state.ledger.state_at(target.id, candidate.eta)
    value = 0.0
    if candidate.kind == "expand":
        value += target.production * remaining * (1.05 if state.is_static(target) else 0.82)
        value -= max(0.0, garrison_at_eta - target.ships) * 0.10
    elif candidate.kind == "attack":
        value += target.production * remaining * 1.35
        value += target.ships * 0.22
        if owner_at_eta not in (-1, state.player):
            value += 12.0
    elif candidate.kind == "comet":
        value += target.production * min(remaining, 22.0) * 0.70
        value -= candidate.eta * 0.35

    if policy is not None:
        value += policy["indirect_value"].get(target.id, 0.0) * min(remaining, 28.0) * 0.07

    enemy_eta, enemy_ships, _ = state.enemy_reach(target)
    race_margin = enemy_eta - candidate.eta
    if race_margin >= 9.0:
        value += min(18.0, race_margin * 1.1)
    elif race_margin <= 2.0:
        value -= min(45.0, 12.0 + enemy_ships * 0.12)

    value -= candidate.ships * 0.06
    return value


def _build_single_capture_candidate(
    state,
    target,
    source,
    available,
    kind,
    planned_commitments=None,
    policy=None,
):
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

    needed = state.min_ships_to_own_at(
        target.id,
        eta,
        planned_commitments=planned_commitments,
        upper_bound=source_available,
    )
    if needed <= 0:
        return None
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

    needed = state.min_ships_to_own_at(
        target.id,
        eta,
        planned_commitments=planned_commitments,
        upper_bound=source_available,
    )
    if needed <= 0:
        return None
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

    score = _candidate_score(state, target, send, eta, kind, policy=policy)
    return Candidate(kind, score, target.id, ((source.id, angle, send),), eta, send, kind)


def _build_multi_attack_candidate(state, target, available, planned_commitments=None, policy=None):
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

        total_cap = sum(source_available for _, _, _, source_available in options)
        needed = state.min_ships_to_own_at(
            target.id,
            target_turn,
            planned_commitments=planned_commitments,
            upper_bound=total_cap,
        )
        if needed <= 0:
            continue
        needed += 4
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
        score = _candidate_score(
            state, target, committed, float(target_turn), "attack", policy=policy
        )
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
    if state.is_static(target):
        priority += 7.0
        if owner == -1 and state.step < 90:
            priority += target.production * 4.0
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


def _generate_capture_candidates(
    state,
    available,
    claimed_targets,
    planned_commitments=None,
    policy=None,
):
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
                state,
                target,
                source,
                available,
                kind,
                planned_commitments=planned_commitments,
                policy=policy,
            )
            if candidate is not None:
                candidates.append(candidate)

        if kind == "attack" and target.production >= 4 and state.step > OPENING_END_STEP:
            candidate = _build_multi_attack_candidate(
                state,
                target,
                available,
                planned_commitments=planned_commitments,
                policy=policy,
            )
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


def _generate_snipe_candidates(state, available, planned_commitments, policy):
    if not USE_SNIPES:
        return []
    candidates = []
    targets = [
        p
        for p in state.neutral_planets + state.comet_planets
        if p.owner == -1 and p.ships <= 28
    ]
    for target in targets:
        arrivals = state.ledger.arrivals.get(target.id, {})
        enemy_turns = sorted(
            turn
            for turn, by_owner in arrivals.items()
            if turn <= SNIPE_HORIZON
            and any(owner != state.player and ships > 0 for owner, ships in by_owner.items())
        )
        if not enemy_turns:
            continue

        for enemy_turn in enemy_turns[:3]:
            sync_turn = int(enemy_turn) + 1
            best_for_target = None
            for source in sorted(
                [p for p in state.my_planets if available.get(p.id, 0) >= PARTIAL_SOURCE_MIN_SHIPS],
                key=lambda p: dist((p.x, p.y), (target.x, target.y)),
            )[:5]:
                source_cap = available[source.id]
                part = _aligned_part_for_turn(source, target, sync_turn, source_cap, state)
                if part is None:
                    continue
                source_id, angle, max_send, eta = part
                if eta > sync_turn + 1.0:
                    continue
                need = state.min_ships_to_own_by(
                    target.id,
                    sync_turn,
                    state.player,
                    arrival_turn=eta,
                    planned_commitments=planned_commitments,
                    upper_bound=max_send,
                )
                if need <= 0 or need > max_send:
                    continue
                if max_send > max(need + 14, int(need * 2.25) + 2):
                    continue
                send = max_send
                score = _candidate_score(
                    state, target, send, float(sync_turn), "snipe", policy=policy
                )
                score += 42.0 + target.production * 8.0 - max(0, sync_turn - enemy_turn) * 3.0
                candidate = Candidate(
                    "snipe",
                    score,
                    target.id,
                    ((source_id, angle, send),),
                    float(sync_turn),
                    send,
                    "enemy_arrival_snipe",
                )
                if best_for_target is None or candidate.score > best_for_target.score:
                    best_for_target = candidate
            if best_for_target is not None:
                candidates.append(best_for_target)
    return candidates


def _generate_recapture_candidates(state, available, planned_commitments, policy, protected_targets):
    if not USE_RECAPTURE:
        return []
    candidates = []
    for target in state.my_planets:
        if target.id in protected_targets:
            continue
        fall_turn = state.ledger.first_not_owned_turn(
            target.id, state.player, DEFENSE_HORIZON
        )
        if fall_turn is None or fall_turn > DEFENSE_HORIZON:
            continue

        for source in sorted(
            [p for p in state.my_planets if p.id != target.id and available.get(p.id, 0) >= PARTIAL_SOURCE_MIN_SHIPS],
            key=lambda p: dist((p.x, p.y), (target.x, target.y)),
        )[:5]:
            source_cap = available[source.id]
            for desired_turn in range(fall_turn + 1, min(fall_turn + RECAPTURE_WINDOW, DEFENSE_HORIZON) + 1):
                part = _aligned_part_for_turn(source, target, desired_turn, source_cap, state)
                if part is None:
                    continue
                source_id, angle, max_send, eta = part
                if eta <= fall_turn or eta > desired_turn + 1.0:
                    continue
                need = state.min_ships_to_own_by(
                    target.id,
                    desired_turn,
                    state.player,
                    arrival_turn=eta,
                    planned_commitments=planned_commitments,
                    upper_bound=max_send,
                )
                if need <= 0 or need > max_send:
                    continue
                if max_send > max(need + 16, int(need * 1.9) + 3):
                    continue
                send = max_send
                saved_turns = max(1.0, state.turns_left() - desired_turn)
                score = (
                    390.0
                    + target.production * saved_turns * 0.62
                    + target.ships * 0.35
                    - send * 0.55
                    - desired_turn * 1.8
                )
                if policy is not None:
                    score += policy["indirect_value"].get(target.id, 0.0) * 5.0
                candidates.append(
                    Candidate(
                        "recapture",
                        score,
                        target.id,
                        ((source_id, angle, send),),
                        float(desired_turn),
                        send,
                        "retake_after_fall",
                    )
                )
                break
    return candidates


def _generate_staging_candidates(state, available):
    if not USE_STAGING:
        return []
    if len(state.my_planets) < 3 or state.step > ENDGAME_STEP:
        return []
    objectives = state.enemy_planets or state.neutral_planets
    if not objectives:
        return []

    safe_fronts = [
        p
        for p in state.my_planets
        if state.ledger.first_not_owned_turn(p.id, state.player, DEFENSE_HORIZON) is None
    ]
    if len(safe_fronts) < 2:
        return []

    frontier_distance = {
        p.id: _nearest_distance_to_set((p.x, p.y), objectives) for p in state.my_planets
    }
    front = min(safe_fronts, key=lambda p: (frontier_distance[p.id], -p.production))
    candidates = []
    for source in sorted(state.my_planets, key=lambda p: -frontier_distance[p.id]):
        if source.id == front.id or available.get(source.id, 0) < STAGING_MIN_SHIPS:
            continue
        if frontier_distance[source.id] < frontier_distance[front.id] * 1.22:
            continue
        send = int(available[source.id] * (0.68 if state.num_players >= 4 else 0.58))
        if send < STAGING_MIN_SHIPS:
            continue
        intercept = _validated_intercept(source, front, send, state, STAGING_MAX_ETA)
        if intercept is None:
            continue
        angle, eta = intercept
        if eta > STAGING_MAX_ETA:
            continue
        score = 24.0 + send * 0.28 + (frontier_distance[source.id] - frontier_distance[front.id]) * 0.7
        candidates.append(
            Candidate(
                "stage",
                score,
                front.id,
                ((source.id, angle, send),),
                eta,
                send,
                "rear_to_front",
            )
        )
        break
    return candidates


def _generate_crash_exploit_candidates(state, available, planned_commitments, policy):
    if not USE_CRASH_EXPLOITS:
        return []
    if state.num_players < 4:
        return []
    candidates = []
    for target in [p for p in state.planets if p.owner != state.player]:
        arrivals = state.ledger.arrivals.get(target.id, {})
        for turn, by_owner in arrivals.items():
            enemy_forces = [
                ships
                for owner, ships in by_owner.items()
                if owner not in (-1, state.player) and ships >= 6
            ]
            if len(enemy_forces) < 2 or turn > 45:
                continue
            enemy_forces.sort(reverse=True)
            if enemy_forces[0] - enemy_forces[1] > 10:
                continue
            desired_turn = int(turn) + 1
            for source in sorted(
                [p for p in state.my_planets if available.get(p.id, 0) >= PARTIAL_SOURCE_MIN_SHIPS],
                key=lambda p: dist((p.x, p.y), (target.x, target.y)),
            )[:4]:
                part = _aligned_part_for_turn(source, target, desired_turn, available[source.id], state)
                if part is None:
                    continue
                source_id, angle, max_send, eta = part
                need = state.min_ships_to_own_by(
                    target.id,
                    desired_turn,
                    state.player,
                    arrival_turn=eta,
                    planned_commitments=planned_commitments,
                    upper_bound=max_send,
                )
                if need <= 0 or need > max_send:
                    continue
                if max_send > max(need + 12, int(need * 1.8) + 2):
                    continue
                send = max_send
                score = _candidate_score(
                    state, target, send, desired_turn, "attack", policy=policy
                ) + 38.0
                candidates.append(
                    Candidate(
                        "crash",
                        score,
                        target.id,
                        ((source_id, angle, send),),
                        float(desired_turn),
                        send,
                        "enemy_crash_cleanup",
                    )
                )
                break
    return candidates


def _apply_candidate(candidate, available, moves, planned_commitments=None, player=None):
    if len(moves) + len(candidate.parts) > MAX_MOVES:
        return False
    for source_id, _, ships in candidate.parts:
        if available.get(source_id, 0) < ships or ships <= 0:
            return False
    total_ships = 0
    for source_id, angle, ships in candidate.parts:
        moves.append([int(source_id), float(angle), int(ships)])
        available[source_id] -= int(ships)
        total_ships += int(ships)
    if planned_commitments is not None and player is not None and total_ships > 0:
        planned_commitments.setdefault(candidate.target_id, []).append(
            (int(math.ceil(candidate.eta)), player, total_ships)
        )
    return True


def _selection_threshold(state, candidate):
    if candidate.kind in ("defend", "evacuate", "recapture"):
        return -9999.0
    if candidate.kind in ("snipe", "crash"):
        return 8.0
    if candidate.kind == "stage":
        return 20.0
    if state.step > ENDGAME_STEP:
        return -8.0
    if state.step < 70 and candidate.kind == "expand":
        return 6.0
    if candidate.kind == "attack":
        return 18.0
    if candidate.kind == "comet":
        return 8.0
    return 10.0


def _position_at_relative_turn(state, planet, relative_turn):
    return predict_planet_position(
        planet, state.step, state.step + float(relative_turn), state.obs
    )


def _opening_target_pool(state, policy):
    targets = [p for p in state.planets if p.owner != state.player and p.id not in state.comet_ids]
    def score(target):
        my_t, enemy_t = _policy_reaction_times(policy, target.id)
        value = target.production * 32.0 - target.ships * 0.85
        value -= _nearest_owned_distance(state, target) * 0.18
        if state.is_static(target):
            value += target.production * 10.0 + 12.0
        else:
            value -= 10.0
            if target.production <= 2:
                value -= 10.0
        if target.owner != -1:
            value += target.production * 12.0 + target.ships * 0.25
        if enemy_t < my_t - 1.0:
            value -= 55.0
        elif enemy_t - my_t >= 6.0:
            value += target.production * 5.0
        value += policy["indirect_value"].get(target.id, 0.0) * 8.0
        return value

    targets.sort(key=score, reverse=True)
    return targets[:OPENING_PLAN_TARGETS]


def _opening_launch_option(state, src_planet, ref_ships, ref_prod, ref_time, target, remaining_steps):
    base_need = int(math.ceil(target.ships + CAPTURE_BUFFER))
    if target.owner not in (-1, state.player):
        base_need += target.production * 4
    if ref_prod <= 0 and ref_ships < base_need:
        return None

    if ref_ships >= base_need:
        earliest = ref_time
    else:
        earliest = ref_time + math.ceil((base_need - ref_ships) / max(1, ref_prod))

    best = None
    for extra in range(OPENING_PLAN_WAIT + 1):
        launch_t = int(math.ceil(earliest + extra))
        if launch_t >= remaining_steps:
            break
        fleet = int(ref_ships + ref_prod * max(0.0, launch_t - ref_time))
        if fleet < base_need:
            continue
        speed = compute_fleet_speed(fleet)
        source_xy = _position_at_relative_turn(state, src_planet, launch_t)
        target_xy = _target_position(target.id, state.step, state.step + launch_t, state.obs)
        if target_xy is None:
            continue
        eta = max(1.0, dist(source_xy, target_xy) / speed)
        for _ in range(5):
            target_xy = _target_position(
                target.id, state.step, state.step + launch_t + eta, state.obs
            )
            if target_xy is None:
                break
            eta = max(1.0, dist(source_xy, target_xy) / speed)
        else:
            cap_t = launch_t + eta
            if cap_t >= remaining_steps:
                continue
            if path_crosses_sun(source_xy, target_xy):
                continue
            need = int(math.ceil(target.ships + CAPTURE_BUFFER))
            if target.owner not in (-1, state.player):
                need += int(math.ceil(target.production * cap_t)) + 2
            if fleet < need:
                continue
            key = (cap_t, -fleet)
            if best is None or key < best[0]:
                best = (
                    key,
                    {
                        "launch_t": launch_t,
                        "send": fleet,
                        "need": need,
                        "eta": eta,
                        "cap_t": cap_t,
                    },
                )
            if extra > 5 and best is not None and cap_t > best[1]["cap_t"] + 1.0:
                break
    return None if best is None else best[1]


def _opening_evaluate_plan(state, plan, policy):
    remaining_steps = max(1, state.turns_left())
    sources = {
        planet.id: (
            max(0, int(planet.ships - policy["reserve"].get(planet.id, 0))),
            int(planet.production),
            0.0,
        )
        for planet in state.my_planets
    }
    value = 0.0
    moves = []
    used_targets = set()
    for src_id, target_id in plan:
        if src_id not in sources or target_id in used_targets or src_id == target_id:
            return None
        src_planet = state.planet_by_id.get(src_id)
        target = state.planet_by_id.get(target_id)
        if src_planet is None or target is None:
            return None
        ref_ships, ref_prod, ref_time = sources[src_id]
        launch = _opening_launch_option(
            state, src_planet, ref_ships, ref_prod, ref_time, target, remaining_steps
        )
        if launch is None:
            return None
        enemy_eta, _, _ = state.enemy_reach(target)
        if target.owner == -1 and enemy_eta < launch["cap_t"] - 0.5:
            return None
        cap_t = launch["cap_t"]
        turns_profit = max(0.0, remaining_steps - cap_t)
        target_value = target.production * turns_profit
        if state.is_static(target):
            target_value *= 1.22
        elif state.step < 35 and target.production <= 2:
            target_value *= 0.72
        if target.owner not in (-1, state.player):
            target_value *= 1.55
        target_value += policy["indirect_value"].get(target.id, 0.0) * turns_profit * 0.10
        value += target_value - launch["send"] * 0.34 - cap_t * 0.45
        moves.append(
            {
                "src_id": src_id,
                "target_id": target_id,
                "launch_t": launch["launch_t"],
                "send": launch["send"],
                "cap_t": cap_t,
            }
        )
        sources[src_id] = (0, ref_prod, float(launch["launch_t"]))
        sources[target_id] = (
            max(0, int(launch["send"] - launch["need"])),
            int(target.production),
            float(cap_t),
        )
        used_targets.add(target_id)
    return {"value": value, "moves": moves, "plan": plan}


def _opening_planner_moves(state, policy, deadline):
    if not USE_OPENING_PLANNER:
        return None
    if time.perf_counter() >= deadline - SPECULATIVE_TIME_MARGIN:
        return None
    if state.step >= OPENING_PLANNER_LIMIT:
        return None
    if state.num_players != 2 or len(state.my_planets) > OPENING_PLAN_MAX_MY_PLANETS:
        return None
    for planet in state.my_planets:
        fall = state.ledger.first_not_owned_turn(
            planet.id, state.player, OPENING_PLAN_DEFENSE_GATE
        )
        if fall is not None:
            return None

    targets = _opening_target_pool(state, policy)
    if not targets:
        return None
    if len(state.my_planets) == 1:
        depth = OPENING_PLAN_DEPTH_ONE
    elif len(state.my_planets) <= 3:
        depth = OPENING_PLAN_DEPTH_FEW
    else:
        depth = OPENING_PLAN_DEPTH_MANY

    stop_at = min(deadline, time.perf_counter() + OPENING_PLANNER_BUDGET)
    beam = [{"plan": [], "value": 0.0, "moves": []}]
    best = None
    initial_sources = {planet.id for planet in state.my_planets}

    for _ in range(depth):
        if time.perf_counter() >= stop_at:
            break
        expanded = []
        for entry in beam:
            if time.perf_counter() >= stop_at:
                break
            used_targets = {target_id for _, target_id in entry["plan"]}
            sources = initial_sources | used_targets
            for src_id in sources:
                for target in targets:
                    if target.id in used_targets or target.id == src_id:
                        continue
                    plan = entry["plan"] + [(src_id, target.id)]
                    evaluated = _opening_evaluate_plan(state, plan, policy)
                    if evaluated is not None:
                        expanded.append(evaluated)
        if not expanded:
            break
        expanded.sort(key=lambda item: item["value"], reverse=True)
        beam = expanded[:OPENING_PLAN_BEAM]
        if best is None or beam[0]["value"] > best["value"]:
            best = beam[0]

    if best is None or not best["moves"] or best["value"] <= 0:
        return None

    available = dict(policy["attack_budget"])
    planned = {}
    moves = []
    for commit in best["moves"]:
        if commit["launch_t"] > 0:
            continue
        source = state.planet_by_id.get(commit["src_id"])
        target = state.planet_by_id.get(commit["target_id"])
        if source is None or target is None:
            continue
        send = min(int(commit["send"]), available.get(source.id, 0))
        if send <= 0:
            continue
        needed = state.min_ships_to_own_at(
            target.id,
            commit["cap_t"],
            planned_commitments=planned,
            upper_bound=send,
        )
        if needed <= 0 or needed > send:
            continue
        intercept = _validated_intercept(source, target, send, state)
        if intercept is None:
            continue
        angle, eta = intercept
        moves.append([int(source.id), float(angle), int(send)])
        available[source.id] -= send
        planned.setdefault(target.id, []).append((int(math.ceil(eta)), state.player, send))
        if len(moves) >= MAX_MOVES:
            break
    return moves or None


def _copy_planned_commitments(planned_commitments):
    return {
        target_id: list(arrivals)
        for target_id, arrivals in (planned_commitments or {}).items()
    }


def _deep_planner_select(state, available, claimed_targets, planned_commitments, policy, deadline):
    if not USE_DEEP_PLANNER:
        return []
    now = time.perf_counter()
    if now >= deadline - SPECULATIVE_TIME_MARGIN:
        return []
    if state.turns_left() <= 3:
        return []

    stop_at = min(deadline - SPECULATIVE_TIME_MARGIN * 0.55, now + PLANNER_BUDGET)
    beam = [
        {
            "available": dict(available),
            "claimed": set(claimed_targets),
            "planned": _copy_planned_commitments(planned_commitments),
            "picks": (),
            "parts": 0,
            "value": 0.0,
        }
    ]
    best = beam[0]

    for _ in range(PLANNER_MAX_PICKS):
        if time.perf_counter() >= stop_at:
            break
        expanded = []
        for entry in beam:
            if time.perf_counter() >= stop_at:
                break
            candidates = _generate_capture_candidates(
                state,
                entry["available"],
                entry["claimed"],
                planned_commitments=entry["planned"],
                policy=policy,
            )
            scored = []
            for candidate in candidates:
                if candidate.kind not in ("expand", "attack", "comet"):
                    continue
                score = _score_candidate_v5(state, candidate, policy)
                bundle_value = score + _planner_projected_value(state, candidate, policy)
                if bundle_value < _selection_threshold(state, candidate):
                    continue
                scored.append((bundle_value, candidate))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)

            for bundle_value, candidate in scored[:PLANNER_TOP_CANDIDATES]:
                if entry["parts"] + len(candidate.parts) > MAX_MOVES:
                    continue
                next_available = dict(entry["available"])
                next_claimed = set(entry["claimed"])
                next_planned = _copy_planned_commitments(entry["planned"])
                test_moves = []
                if not _apply_candidate(
                    candidate,
                    next_available,
                    test_moves,
                    next_planned,
                    state.player,
                ):
                    continue
                next_claimed.add(candidate.target_id)
                next_entry = {
                    "available": next_available,
                    "claimed": next_claimed,
                    "planned": next_planned,
                    "picks": entry["picks"] + (candidate,),
                    "parts": entry["parts"] + len(candidate.parts),
                    "value": entry["value"] + bundle_value,
                }
                expanded.append(next_entry)

        if not expanded:
            break
        expanded.sort(key=lambda item: item["value"], reverse=True)
        beam = expanded[:PLANNER_BEAM]
        if beam[0]["value"] > best["value"]:
            best = beam[0]

    if best["picks"] and best["value"] > 0.0:
        return list(best["picks"])
    return []


def _choose_moves(state, deadline=None):
    if deadline is None:
        deadline = time.perf_counter() + 0.82
    policy = _build_policy(state)
    opening_moves = _opening_planner_moves(state, policy, deadline)
    if opening_moves is not None:
        return opening_moves[:MAX_MOVES]

    available = dict(policy["attack_budget"])
    moves = []
    claimed_targets = set()
    planned_commitments = {}

    defense_candidates = sorted(
        _generate_defense_candidates(state, available),
        key=lambda c: c.score,
        reverse=True,
    )
    protected_targets = set()
    for candidate in defense_candidates:
        if _apply_candidate(
            candidate, available, moves, planned_commitments, state.player
        ):
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
        if _apply_candidate(
            candidate, available, moves, planned_commitments, state.player
        ):
            claimed_targets.add(candidate.target_id)
        if len(moves) >= MAX_MOVES:
            return moves[:MAX_MOVES]

    tactical_candidates = []
    if time.perf_counter() < deadline - SPECULATIVE_TIME_MARGIN:
        tactical_candidates = sorted(
            _generate_snipe_candidates(state, available, planned_commitments, policy)
            + _generate_recapture_candidates(
                state, available, planned_commitments, policy, protected_targets
            )
            + _generate_crash_exploit_candidates(
                state, available, planned_commitments, policy
            ),
            key=lambda c: c.score,
            reverse=True,
        )
    for candidate in tactical_candidates:
        if candidate.target_id in claimed_targets and candidate.kind not in ("recapture",):
            continue
        if candidate.score < _selection_threshold(state, candidate):
            continue
        if _apply_candidate(
            candidate, available, moves, planned_commitments, state.player
        ):
            claimed_targets.add(candidate.target_id)
        if len(moves) >= MAX_MOVES:
            return moves[:MAX_MOVES]

    planner_applied = False
    if time.perf_counter() < deadline - SPECULATIVE_TIME_MARGIN:
        for candidate in _deep_planner_select(
            state, available, claimed_targets, planned_commitments, policy, deadline
        ):
            if candidate.target_id in claimed_targets:
                continue
            if _score_candidate_v5(state, candidate, policy) < _selection_threshold(
                state, candidate
            ):
                continue
            if _apply_candidate(
                candidate, available, moves, planned_commitments, state.player
            ):
                planner_applied = True
                claimed_targets.add(candidate.target_id)
            if len(moves) >= MAX_MOVES:
                return moves[:MAX_MOVES]

    fallback_iterations = 4 if planner_applied else 8
    for _ in range(fallback_iterations):
        candidates = _generate_capture_candidates(
            state,
            available,
            claimed_targets,
            planned_commitments=planned_commitments,
            policy=policy,
        )
        if not candidates:
            break
        candidates.sort(key=lambda c: _score_candidate_v5(state, c, policy), reverse=True)
        chosen = candidates[0]
        if _score_candidate_v5(state, chosen, policy) < _selection_threshold(state, chosen):
            break
        if _apply_candidate(chosen, available, moves, planned_commitments, state.player):
            claimed_targets.add(chosen.target_id)
        else:
            claimed_targets.add(chosen.target_id)
        if len(moves) >= MAX_MOVES:
            break

    if len(moves) < MAX_MOVES and time.perf_counter() < deadline - SPECULATIVE_TIME_MARGIN:
        staging_candidates = sorted(
            _generate_staging_candidates(state, available),
            key=lambda c: c.score,
            reverse=True,
        )
        for candidate in staging_candidates:
            if candidate.score < _selection_threshold(state, candidate):
                continue
            _apply_candidate(candidate, available, moves, planned_commitments, state.player)
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
    global _CURRENT_STATE
    start = time.perf_counter()
    state = GameState(obs)
    _CURRENT_STATE = state
    if not state.my_planets:
        return []
    act_timeout = obs_get(config, "actTimeout", 1.0) if config is not None else 1.0
    deadline = start + min(0.84, max(0.55, float(act_timeout) * 0.82))
    return _choose_moves(state, deadline=deadline)


if __name__ == "__main__":
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["main.py", "random"])
    final = env.steps[-1]
    for i, state in enumerate(final):
        print(f"Player {i}: reward={state.reward}, status={state.status}")
