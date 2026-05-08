import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
SUN_MARGIN = 0.35
ROTATION_RADIUS_LIMIT = 50.0
MAX_FLEET_SPEED = 6.0

CAPTURE_BUFFER = 3
MAX_ETA = 95
MAX_MOVES = 12

_OBS_CACHE = {}


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
        (Planet(*raw) for raw in obs_get(obs, "initial_planets", []) if raw[0] == planet_id),
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


def _target_position(target_planet_id, current_step, future_step, obs):
    planet = _planet_by_id(obs, target_planet_id)
    if planet is None:
        return None
    return predict_planet_position(planet, current_step, future_step, obs)


def _sun_safe_angle(source_xy, target_xy, target_radius):
    direct_angle = math.atan2(target_xy[1] - source_xy[1], target_xy[0] - source_xy[0])
    if not path_crosses_sun(source_xy, target_xy):
        return direct_angle, target_xy

    base_dist = dist(source_xy, target_xy)
    if base_dist <= 1e-9:
        return None

    # If the planet's near edge is visible around the sun, aim at that edge.
    offsets = []
    for mult in (0.7, 1.0, 1.3):
        offsets.extend((target_radius * mult, -target_radius * mult))
    for offset in offsets:
        perp = direct_angle + math.pi / 2.0
        aim = (
            target_xy[0] + math.cos(perp) * offset,
            target_xy[1] + math.sin(perp) * offset,
        )
        if not path_crosses_sun(source_xy, aim):
            return math.atan2(aim[1] - source_xy[1], aim[0] - source_xy[0]), aim

    # Last-ditch small deflections. These only work when the target disk is wide
    # enough to absorb the miss, so verify closest approach to the target body.
    for degrees in (5, -5, 10, -10, 15, -15, 22, -22):
        angle = direct_angle + math.radians(degrees)
        end = (
            source_xy[0] + math.cos(angle) * base_dist,
            source_xy[1] + math.sin(angle) * base_dist,
        )
        if path_crosses_sun(source_xy, end):
            continue
        if point_to_segment_distance(target_xy, source_xy, end) <= target_radius * 0.85:
            return angle, end
    return None


def solve_intercept_angle(source_xy, target_planet_id, num_ships, current_step, obs):
    target = _planet_by_id(obs, target_planet_id)
    if target is None:
        return None

    speed = compute_fleet_speed(num_ships)
    target_xy = _target_position(target_planet_id, current_step, current_step, obs)
    if target_xy is None:
        return None

    eta = max(1.0, dist(source_xy, target_xy) / speed)
    for _ in range(5):
        target_xy = _target_position(target_planet_id, current_step, current_step + eta, obs)
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


def ships_needed_to_capture(target_planet, eta, current_step, obs):
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
            if target_planet.owner == player:
                needed -= fleet.ships
            else:
                needed -= fleet.ships * 0.9
        elif target_planet.owner != -1 and fleet.owner == target_planet.owner:
            needed += fleet.ships
        else:
            needed += fleet.ships * 0.4

    return max(1, int(math.ceil(needed + CAPTURE_BUFFER)))


def _reserve_for_planet(planet, current_step, my_total, enemy_total):
    if planet.id < 0:
        return 0
    if current_step < 35:
        reserve = max(3, int(planet.production * 2))
    else:
        reserve = max(5, int(planet.production * 3))
    if my_total > enemy_total * 1.8 and current_step > 120:
        reserve = max(2, reserve // 2)
    return reserve


def _send_amount_with_speed_bonus(needed, available, distance_to_target, current_step):
    needed = int(max(1, needed))
    if available < needed:
        return 0
    send = needed
    if distance_to_target > 35:
        send = max(send, int(needed * 1.20) + 2)
    if distance_to_target > 55:
        send = max(send, int(needed * 1.45) + 4)
    if distance_to_target > 70:
        send = max(send, int(needed * 1.75) + 6)
    if current_step < 45 and needed <= 12 and available >= needed + 4:
        send = max(send, needed + 4)
    return min(available, send)


def _add_move(moves, available, source_id, angle, ships):
    ships = int(ships)
    if ships <= 0 or available.get(source_id, 0) < ships:
        return False
    moves.append([int(source_id), float(angle), ships])
    available[source_id] -= ships
    return True


def _best_single_source(target, sources, available, current_step, obs):
    best = None
    for source in sources:
        if available.get(source.id, 0) <= 0:
            continue

        source_xy = (source.x, source.y)
        probe = max(available[source.id], target.ships + CAPTURE_BUFFER)
        intercept = solve_intercept_angle(source_xy, target.id, probe, current_step, obs)
        if intercept is None:
            continue
        angle, eta = intercept
        needed = ships_needed_to_capture(target, eta, current_step, obs)
        send = _send_amount_with_speed_bonus(
            needed, available[source.id], dist(source_xy, (target.x, target.y)), current_step
        )
        if send < needed:
            continue
        intercept = solve_intercept_angle(source_xy, target.id, send, current_step, obs)
        if intercept is None:
            continue
        angle, eta = intercept
        needed = ships_needed_to_capture(target, eta, current_step, obs)
        send = _send_amount_with_speed_bonus(
            needed, available[source.id], dist(source_xy, (target.x, target.y)), current_step
        )
        if send < needed:
            continue

        value = (target.production * 45.0) / (eta + send * 0.35 + 1.0)
        if target.owner == -1:
            value += target.production * 1.5
        if target.id in set(obs_get(obs, "comet_planet_ids", []) or []):
            value -= 2.0
        if best is None or value > best[0]:
            best = (value, source, angle, send, eta, needed)
    return best


def _incoming_to_planet(target, player, current_step, obs, horizon=35):
    incoming_enemy = 0
    incoming_friend = 0
    first_enemy_eta = None
    for fleet in as_fleets(obs):
        eta = _fleet_arrival_eta(fleet, target, current_step, obs, horizon)
        if eta is None:
            continue
        if fleet.owner == player:
            incoming_friend += fleet.ships
        else:
            incoming_enemy += fleet.ships
            if first_enemy_eta is None or eta < first_enemy_eta:
                first_enemy_eta = eta
    return incoming_enemy, incoming_friend, first_enemy_eta


def _defense_pass(moves, my_planets, available, current_step, obs):
    player = obs_get(obs, "player", 0)
    for target in sorted(my_planets, key=lambda p: p.ships):
        enemy_ships, friendly_ships, enemy_eta = _incoming_to_planet(
            target, player, current_step, obs, horizon=32
        )
        if enemy_eta is None:
            continue
        future_garrison = target.ships + target.production * enemy_eta + friendly_ships
        if future_garrison >= enemy_ships + 2:
            continue
        needed = int(math.ceil(enemy_ships - future_garrison + 4))

        reinforcers = sorted(
            [p for p in my_planets if p.id != target.id and available.get(p.id, 0) > 0],
            key=lambda p: dist((p.x, p.y), (target.x, target.y)),
        )
        for source in reinforcers:
            if needed <= 0:
                break
            send = min(available[source.id], needed)
            intercept = solve_intercept_angle((source.x, source.y), target.id, send, current_step, obs)
            if intercept is None:
                continue
            angle, eta = intercept
            if eta > enemy_eta + 1.0:
                continue
            if _add_move(moves, available, source.id, angle, send):
                needed -= send


def _comet_pass(moves, my_planets, available, current_step, obs):
    comet_ids = set(obs_get(obs, "comet_planet_ids", []) or [])
    if not comet_ids:
        return
    comets = [
        p
        for p in as_planets(obs)
        if p.id in comet_ids and p.owner != obs_get(obs, "player", 0) and p.ships <= 18
    ]
    for target in sorted(comets, key=lambda p: (p.ships, -p.production))[:3]:
        best = _best_single_source(target, my_planets, available, current_step, obs)
        if best is None:
            continue
        _, source, angle, send, eta, _ = best
        if eta > 18:
            continue
        _add_move(moves, available, source.id, angle, send)
        if len(moves) >= MAX_MOVES:
            return


def _expansion_pass(moves, my_planets, targets, available, current_step, obs):
    scored = []
    for target in targets:
        best = _best_single_source(target, my_planets, available, current_step, obs)
        if best is None:
            continue
        value, source, angle, send, eta, needed = best
        value += target.production * 4.0
        value -= target.ships * 0.08
        value -= eta * 0.04
        scored.append((value, target, source, angle, send, eta, needed))

    scored.sort(reverse=True, key=lambda item: item[0])
    launched = 0
    for _, target, source, angle, send, eta, needed in scored:
        if launched >= 4 or len(moves) >= MAX_MOVES:
            break
        if target.owner != -1:
            continue
        if ships_needed_to_capture(target, eta, current_step, obs) > send:
            continue
        if _add_move(moves, available, source.id, angle, send):
            launched += 1


def _attack_pass(moves, my_planets, enemy_planets, available, current_step, obs):
    if len(moves) >= MAX_MOVES:
        return

    enemy_planets = sorted(
        enemy_planets,
        key=lambda p: (p.production, -p.ships),
        reverse=True,
    )

    for target in enemy_planets[:8]:
        options = []
        for source in my_planets:
            if available.get(source.id, 0) <= 0:
                continue
            source_xy = (source.x, source.y)
            probe = available[source.id]
            intercept = solve_intercept_angle(source_xy, target.id, probe, current_step, obs)
            if intercept is None:
                continue
            angle, eta = intercept
            if eta > 70:
                continue
            options.append((eta, source, angle, dist(source_xy, (target.x, target.y))))

        if not options:
            continue
        options.sort(key=lambda item: item[0])

        committed = []
        committed_ships = 0
        attack_eta = 0.0
        for eta, source, angle, distance_to_target in options[:4]:
            attack_eta = max(attack_eta, eta)
            needed = ships_needed_to_capture(target, attack_eta, current_step, obs) + 2
            send = _send_amount_with_speed_bonus(
                min(needed - committed_ships, available[source.id]),
                available[source.id],
                distance_to_target,
                current_step,
            )
            if send <= 0:
                continue
            committed.append((source, angle, send))
            committed_ships += send
            if committed_ships >= needed:
                break

        if not committed:
            continue
        needed = ships_needed_to_capture(target, attack_eta, current_step, obs) + 2
        if committed_ships < needed:
            continue
        for source, angle, send in committed:
            _add_move(moves, available, source.id, angle, send)
        return


def _total_ships(planets, fleets, owner):
    return sum(p.ships for p in planets if p.owner == owner) + sum(
        f.ships for f in fleets if f.owner == owner
    )


def agent(obs, config=None):
    del config
    _prepare_obs_cache(obs)
    current_step = int(obs_get(obs, "step", 0))
    player = obs_get(obs, "player", 0)
    planets = as_planets(obs)
    fleets = as_fleets(obs)
    comet_ids = set(obs_get(obs, "comet_planet_ids", []) or [])

    my_planets = [p for p in planets if p.owner == player]
    if not my_planets:
        return []

    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
    neutral_planets = [p for p in planets if p.owner == -1 and p.id not in comet_ids]
    my_total = _total_ships(planets, fleets, player)
    enemy_total = sum(_total_ships(planets, fleets, owner) for owner in range(4) if owner != player)

    available = {}
    for planet in my_planets:
        reserve = _reserve_for_planet(planet, current_step, my_total, enemy_total)
        if planet.id in comet_ids:
            reserve = 0
        available[planet.id] = max(0, int(planet.ships - reserve))

    moves = []
    _defense_pass(moves, my_planets, available, current_step, obs)
    if len(moves) < MAX_MOVES:
        _comet_pass(moves, my_planets, available, current_step, obs)

    if len(moves) < MAX_MOVES:
        neutral_candidates = sorted(
            neutral_planets,
            key=lambda p: (p.production * 8.0 - p.ships, p.production),
            reverse=True,
        )[:14]
        _expansion_pass(
            moves, my_planets, neutral_candidates, available, current_step, obs
        )

    if len(moves) < MAX_MOVES and (current_step > 45 or not neutral_planets):
        _attack_pass(moves, my_planets, enemy_planets, available, current_step, obs)

    return moves[:MAX_MOVES]


if __name__ == "__main__":
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["main.py", "random"])
    final = env.steps[-1]
    for i, state in enumerate(final):
        print(f"Player {i}: reward={state.reward}, status={state.status}")
