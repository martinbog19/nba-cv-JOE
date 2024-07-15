import cv2
import numpy as np
from collections import defaultdict

def apply_homography(H,pts):

    assert(H.shape==(3,3))
    assert(pts.shape[0]==2)
    assert(pts.shape[1]>=1)

    tpts = np.zeros(pts.shape)
    for i in range(pts.shape[1]):
        u = H[0][0]*pts[0][i] + H[0][1]*pts[1][i] + H[0][2]
        v = H[1][0]*pts[0][i] + H[1][1]*pts[1][i] + H[1][2]
        w = H[2][0]*pts[0][i] + H[2][1]*pts[1][i] + H[2][2]

        x_prime = u/w
        y_prime = v/w

        tpts[0][i] = x_prime
        tpts[1][i] = y_prime

    assert(tpts.shape[0]==2)
    assert(tpts.shape[1]==pts.shape[1])

    return tpts

def project(box, side):

    assert((side == "right") or (side == "left"))

    if side == "right":
        src = np.array([
        [895,280],
        [1280,522], 
        [1,633], 
        [1,325] 
        ]) 

        court = np.array([
        [575, 35],
        [575,  347],
        [306,  347], 
        [306,  35],  
        ]) 

    
    else: 

        src = np.array([
        [1260,338], 
        [1250,660], 
        [90,540], 
        [432,305] 
        ]) 

        court = np.array([
        [306,  35],
        [306,  347],
        [36,347],
        [36,35], 
        ])

    homography, _ = cv2.findHomography(src, court)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    xc = x1 + int((x2 - x1)/2)

    player_pos = np.array([[xc],[y2]])
    projected_pos = apply_homography(homography,player_pos)

    return np.array([projected_pos[0][0], projected_pos[0][1]])

def distance(x1, y1, x2, y2):
    
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def closest_distance(x_player: float, y_player: float, coords: list):
    """
    Get the closest defender distance for a given player
    
    x_player: x position of the player
    y_player: y position of the player
    coords: coordinates of all defense players, list of tuples of size 2
    """
    minimum = np.inf
    for x, y in coords:
        if distance(x_player, y_player, x, y) < minimum:
            minimum = distance(x_player, y_player, x, y)
    
    return minimum

def compute_joe(players_trajectories: dict, players_teams: dict, offense_team: str):
    """
    Compute Joe for every offense players on the sequence

    players_trajectories (dict): trajectories dico
    players_teams (dict): matching dico for players and teams
    offense_team (str): name of the team (as on the players_teams dict)
    """
    
    nb_frames = len(players_trajectories.values()[0]["x"])
    joe = defaultdict(list)

    offense_players = [player for player in players_trajectories.keys() if players_teams[player] == offense_team]
    defense_players = [player for player in players_trajectories.keys() if players_teams[player] != offense_team]

    for t in nb_frames:
        def_coords = [(players_trajectories[player]["x"][t], players_trajectories[player]["y"][t]) for player in defense_players]
        for player in offense_players:
            joe[player].append(closest_distance(players_trajectories[player]["x"][t], players_trajectories[player]["y"][t], def_coords))

    return {key: np.mean(value) for key, value in joe.items()}


def interpolate_points(players_trajectories: dict):

    """
    Modify the player trajectories dict by interpolating the points that are missing
    """

    for player, traj in players_trajectories.items():
        ind_nan = np.isnan(np.array(traj["x"]))
        ind_non_nan = ~ind_nan
        x_interp = np.array(traj["x"]).copy()
        y_interp = np.array(traj["y"]).copy()
        x_interp[ind_nan] = np.interp(np.flatnonzero(ind_nan), np.flatnonzero(ind_non_nan), np.array(traj["x"])[ind_non_nan])
        y_interp[ind_nan] = np.interp(np.flatnonzero(ind_nan), np.flatnonzero(ind_non_nan), np.array(traj["y"])[ind_non_nan])
        players_trajectories[player]["x"] = x_interp
        players_trajectories[player]["y"] = y_interp
    
    return players_trajectories


def discard_errors(players_trajectories: dict, first_positions: dict, tresh: float,):
    
    """
    Remove any wrong predictions by a given treshold
    """
    new_trajectories = players_trajectories.copy()
    new_trajectories = { key : {"x": [first_positions[key]["x"][0]], "y": [first_positions[key]["y"][0]]} for key, _ in new_trajectories.items()}
    nb_frames = len(list(players_trajectories.values())[0]["x"])
    for player, traj in players_trajectories.items():
        step_forward = 0
        for t in range(nb_frames-1):
            t = t + step_forward
            if t >= nb_frames-1: # break if we are over the length of the sequence
                break
            close = False
            step = 0
            while not close:
                ## get the tamp list with only nans
                tamp_x, tamp_y = np.full(2 + step, np.nan), np.full(2 + step, np.nan)

                ## fix the first and last point
                tamp_x[0], tamp_y[0], tamp_x[-1], tamp_y[-1] = traj["x"][t], traj["y"][t], traj["x"][t + 1+step], traj["y"][t +1+step]

                ## interpolate the list
                ind_non_nan = np.arange(len(tamp_x))[~np.isnan(tamp_x)]
                tamp_int_x = np.interp(np.arange(len(tamp_x)), ind_non_nan, tamp_x[ind_non_nan])
                tamp_int_y = np.interp(np.arange(len(tamp_y)), ind_non_nan, tamp_y[ind_non_nan])
                

                ## test if the next point is close enough
                if distance(tamp_int_x[0], tamp_int_y[0], tamp_int_x[1], tamp_int_y[1]) <= tresh:
                    close = True
                    new_trajectories[player]["x"].append(traj["x"][t+1+step])
                    new_trajectories[player]["y"].append(traj["y"][t+1+step])

                ## else we add a np.nan and look at the next point
                else:
                    new_trajectories[player]["x"].append(np.nan)
                    new_trajectories[player]["y"].append(np.nan)
                    step_forward += 1
                    step += 1
                
                ## break the loop if we are at the end of the sequences
                if t + step >= nb_frames - 1:
                    break
        
        assert(len(new_trajectories[player]["x"]) == nb_frames)

    return new_trajectories