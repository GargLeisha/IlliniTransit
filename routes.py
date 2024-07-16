from datetime import datetime, timedelta
import networkx as nx
import pandas as pd
import math
import heapq
import pickle
import os
import requests
from dotenv import load_dotenv


load_dotenv()






# Path to the serialized graph file
GRAPH_FILE = 'graph.pkl'


def initialize_graph():
 if os.path.exists(GRAPH_FILE):
     print("Loading existing graph from file")
     with open(GRAPH_FILE, 'rb') as f:
         G = pickle.load(f)
 else:
     print("Creating a new graph")
     stops = pd.read_csv('stops.txt')
     stop_times = pd.read_csv('stop_times.txt')
     trips = pd.read_csv('trips.txt')


     G = nx.MultiDiGraph()


     G.add_nodes_from(stops.set_index('stop_id').to_dict('index').items())
     trip_stop_times = stop_times.merge(trips, on='trip_id')


     for trip_id, trip_data in trip_stop_times.groupby('trip_id'):
         sorted_stops = trip_data.sort_values('stop_sequence')
         stop_ids = sorted_stops['stop_id'].tolist()
         for i in range(len(stop_ids) - 1):
             stop1 = stop_ids[i]
             stop2 = stop_ids[i + 1]
             start_time = parse_time(sorted_stops.iloc[i]["arrival_time"])
             end_time = parse_time(sorted_stops.iloc[i + 1]["arrival_time"])
           
             travel_time = (end_time - start_time).total_seconds() / 60
             G.add_edge(stop1, stop2, trip_id=trip_id, mode="transit", start_time=sorted_stops.iloc[i]["arrival_time"], end_time=sorted_stops.iloc[i + 1]["arrival_time"], weight=travel_time)


     nodes = list(G.nodes(data=True))
     count = 0
     for i in range(len(nodes)):
         for j in range(i + 1, len(nodes)):
             node1, data1 = nodes[i]
             node2, data2 = nodes[j]
             count += 2
             print(f"Getting walk time {count} between {node1} and {node2}")
             walk_time = get_walk_time_from_osrm(data1['stop_lat'], data1['stop_lon'], data2['stop_lat'], data2['stop_lon'])
             G.add_edge(node1, node2, mode="walk", weight=walk_time)
             G.add_edge(node2, node1, mode="walk", weight=walk_time)
  
     with open(GRAPH_FILE, 'wb') as f:
         pickle.dump(G, f)


 return G

def update_transit_edges_with_service_id(G, trip_data):
  """Update all transit edges in the graph with the corresponding service_id."""
  for u, v, key, edge_data in G.edges(keys=True, data=True):
      if edge_data['mode'] == 'transit':
          trip_id = edge_data['trip_id']
          service_id = trip_data.loc[trip_id, 'service_id']
          G[u][v][key]['service_id'] = service_id
  print("All transit edges updated with service_id")


def add_service_dates_to_edges(G, service_dates):
  for u, v, key, edge_data in G.edges(keys=True, data=True):
      if edge_data['mode'] == 'transit':
          service_id = edge_data['service_id']
          if service_id in service_dates:
              G[u][v][key]['service_dates'] = service_dates[service_id]
  print("All transit edges updated with service_dates")


def get_walk_time_from_osrm(lat1, lon1, lat2, lon2):
  url = f"http://localhost:5001/route/v1/walking/{lon1},{lat1};{lon2},{lat2}?overview=false"
  try:
      response = requests.get(url)
      data = response.json()
      if 'routes' in data and len(data['routes']) > 0:
          duration = data['routes'][0]['duration']
          return duration / 60  # Convert seconds to minutes
      else:
          return float('inf')  # Return infinity if no route is found
  except Exception as e:
      print(f"Error: {e}")
      return float('inf')


def parse_time(time_str):
  if isinstance(time_str, pd.Timestamp):
      return time_str


  hours, minutes, seconds = map(int, time_str.split(':'))
  if hours >= 24:
      hours -= 24
      extra_day = pd.Timedelta(days=1)
  else:
      extra_day = pd.Timedelta(0)
  time_of_day = pd.to_datetime(f'{hours:02}:{minutes:02}:{seconds:02}', format='%H:%M:%S')
  return time_of_day + extra_day


def print_connected_nodes(G, node):
  if node in G:
      connected_nodes = list(G.neighbors(node))
      print(f"Nodes connected to {node}: {connected_nodes}")
  else:
      print(f"Node {node} is not in the graph.")


def to_rad(degrees):
  return degrees * math.pi / 180


def calculate_walk_time(lat1, long1, lat2, long2):
  R = 6371  # Radius of the Earth in km
  dLat = to_rad(lat2 - lat1)
  dLong = to_rad(long2 - long1)
  a = math.sin(dLat / 2) ** 2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dLong / 2) ** 2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  d = R * c
  return (d / 4) * 60  # Time in minutes


def get_user_location():
  try:
      response = requests.get('https://ipinfo.io')
      data = response.json()
      loc = data.get('loc', '')
      if loc:
          latitude, longitude = loc.split(',')
          return float(latitude), float(longitude)
      else:
          return None, None
  except Exception as e:
      print(f"Error: {e}")
      return None, None


def add_user_location_to_graph(G):
  user_lat, user_lon = get_user_location()
  if user_lat is not None and user_lon is not None:
      user_node = 'user_location'
      G.add_node(user_node, stop_lat=user_lat, stop_lon=user_lon)
      nodes = list(G.nodes(data=True))
      for node, data in nodes:
          if node != user_node:
              walk_time = calculate_walk_time(user_lat, user_lon, data['stop_lat'], data['stop_lon'])
              if walk_time < 30:
                  G.add_edge(user_node, node, mode="walk", weight=walk_time)
                  G.add_edge(node, user_node, mode="walk", weight=walk_time)
                  find_walk_times(G, user_node, node)
      return user_node
  else:
      print("Unable to retrieve user location.")
      return None


def calculate_wait_time(end_time, start_time):
    end_time = parse_time(end_time)
    start_time = parse_time(start_time)
    if start_time < end_time:
        start_time += pd.Timedelta(days=1)
    wait_time = (start_time - end_time).total_seconds() / 60  # Wait time in minutes
    return wait_time


    
def find_earlier_stop(G, path, start_times, end_times, wait_times, durations, modes, trip_ids, routes, start_node, current_time):
   print("starting...")
   trip_id_universal = ""
   visited_nodes = set()
   
   stop_times = pd.read_csv('stop_times.txt', header=None, names=[
       'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'stop_name', 'pickup_type', 'drop_off_type', 'shape_dist_traveled'
   ], dtype={
       'trip_id': str,
       'arrival_time': str,
       'departure_time': str,
       'stop_id': str,
       'stop_sequence': str,  # Load as str to handle mixed types
       'stop_name': str,
       'pickup_type': str,
       'drop_off_type': str,
       'shape_dist_traveled': str
   }, low_memory=False)
  
   # Convert specific columns to appropriate types
   stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce')
  
   print("csv loaded without errors")
  
   for i in range(len(modes)):
       if modes[i] == "transit":
           trip_id_local = trip_ids[i]
           if trip_id_local != trip_id_universal:
               trip_id_universal = trip_id_local
              
               #gives all the terip_data whtere trip_id is equal to the universal one, sorting by stop sequence
               trip_data = stop_times[stop_times['trip_id'] == trip_id_universal].sort_values('stop_sequence')
              
               current_stop_id = path[i]
               if trip_data.empty:
                   continue  # Skip if no trip data is found

               # gets the row for the data of the current stop
               current_stop_data = trip_data[trip_data['stop_id'] == current_stop_id]
            

               if current_stop_data.empty:
                   continue  # Skip if current stop_id is not found

               stop_sequence_index = int(current_stop_data['stop_sequence'].iloc[0])
           
              
              
             

               # Initialize variables to keep track of the best stop
               best_stop_id = None
               best_walk_time = float('inf')

               # Check earlier stops in the sequence
               for j in range(stop_sequence_index - 1, 0, -1):
                   
                   #first we iterate backwards to check all the stop_ids in that stop sequence 
                   earlier_stop_id = trip_data.iloc[j]['stop_id']
                   start_time_str = trip_data.iloc[j]['departure_time']
                   start_time = parse_time(start_time_str)
                  
                   
                   #checks to see if it is a neigbor of the start node
                   if earlier_stop_id in G[start_node]:
                        walk_distance = G[start_node][earlier_stop_id][0]['weight']
                        walk_time = timedelta(minutes=walk_distance)
                        arrival_time = parse_time(current_time) + walk_time

                        if start_time >= arrival_time and walk_distance < best_walk_time:
                            best_stop_id = earlier_stop_id
                            best_walk_time = walk_distance

               # Update path and attributes with the best earlier stop
               if best_stop_id:
                   # Find the sub-path from best_stop_id to current_stop_id
                   sub_path_data = trip_data[(trip_data['stop_sequence'] >= trip_data[trip_data['stop_id'] == best_stop_id]['stop_sequence'].values[0]) &
                                             (trip_data['stop_sequence'] <= stop_sequence_index)]

                   if sub_path_data.empty:
                       continue

                   #makes the stops into a path
                   walk_end_time = (current_time) + timedelta(minutes=best_walk_time)
                   wait_time = calculate_wait_time(walk_end_time, sub_path_data.iloc[0]['departure_time'])

                   sub_path = sub_path_data['stop_id'].tolist()
                   sub_start_times = [parse_time(time_str) for time_str in sub_path_data['arrival_time'].tolist()]
                   sub_end_times = [parse_time(time_str) for time_str in sub_path_data['departure_time'].tolist()]
                   sub_end_times = sub_end_times[1:] + [sub_end_times[0]] 
                   sub_wait_times =  [wait_time] + [0] * (len(sub_path)-1) 
                   sub_durations = [(sub_end_times[k] - sub_start_times[k]).total_seconds() / 60 for k in range(len(sub_path))]
                   sub_modes = ['transit'] * len(sub_path)
                   sub_trip_ids = [trip_id_universal] * len(sub_path)
                   trips = pd.read_csv("trips.txt")
                   service_id = trips.loc[trips["trip_id"] == trip_id_universal, 'service_id'].values[0]
                   sub_routes = [service_id] * len(sub_path)
              
                   walk_start_time = start_times[0]
                   walk_end_time = walk_start_time + timedelta(minutes=best_walk_time)
                   walk_duration = best_walk_time
                   # Update the original path and related attributes
                  
                  
                   path = [start_node] + sub_path + path[i+1:]
                   start_times = [walk_start_time] + sub_start_times + start_times[i+1:]
                   end_times = [walk_end_time] + sub_end_times + end_times[i+1:]
                   wait_times = [0] + sub_wait_times + wait_times[i+1:]
                   durations = [walk_duration] + sub_durations + durations[i+1:]
                   modes = ["walk"] + sub_modes + modes[i+1:]
                   trip_ids = [None] + sub_trip_ids + trip_ids[i+1:]
                   routes = [None] + sub_routes + routes[i+1:]

                   

   
   return path, start_times, end_times, wait_times, durations, modes, trip_ids, routes
    
def find_shortest_path(G, start_node, end_node, current_time_str, current_date_str):
    
  starting_time = parse_time(current_time_str)
  visited = None
  pq = None
  visited = set()

#weight, the node, end time cuz u end ur time at first node when you leave it 
  pq = [(0, start_node)]
  
  #the node, the weight of the edge, the previous node, end time, the wait time, the actual duration, the mode, the trip id, the route
  #this is the dictionary, where the node is the key. Attributes: prev_node, total_time, weight (wait + actual), start_time, end_time, wait_time, duration, mode, trip id, service_id
  shortest_paths = {start_node: (None, 0, 0, starting_time, starting_time, 0, 0, None, None, None)}
  while pq:
      total_time, current_node= heapq.heappop(pq)
      if current_node in visited:
          continue
      visited.add(current_node)
      for neighbor, edge_data in G[current_node].items():
          for key, attributes in edge_data.items():
              if attributes['mode'] == 'transit':
                  service_dates = [str(d).strip() for d in attributes['service_dates']]
                  #should change graph so this is already in right format
                  if current_date_str not in service_dates:
                      continue  # Skip this edge if the service does not run on the current date
                #this weights is just the duration stored in edges 
                  duration = attributes['weight']
                #this start time is when the bus starts 
                  start_time = parse_time(attributes['start_time'])
            
                  wait_time = calculate_wait_time(shortest_paths[current_node][4], start_time)
                  new_total_time = total_time + wait_time + duration
                  
                  if (neighbor not in shortest_paths or new_total_time < shortest_paths[neighbor][1]) and shortest_paths[current_node][4] <= start_time:
                      # prev_node, total_time, weight (wait + actual), start_time, end_time, wait_time, duration, mode, trip id, service_id
                      weight = wait_time + duration
                      end_time = parse_time(attributes["end_time"])
                      mode = "transit"
                      trip_id = attributes['trip_id']  # Get route info
                      service_id = attributes['service_id']
                      shortest_paths[neighbor] = (current_node, new_total_time, weight, start_time, end_time, wait_time, duration, mode, trip_id, service_id )
                      heapq.heappush(pq, (new_total_time, neighbor))


              elif attributes['mode'] == 'walk':
                  first_one = False
                  duration = attributes['weight']
                  new_total_time = shortest_paths[current_node][1] + duration
                  if neighbor not in shortest_paths or new_total_time < shortest_paths[neighbor][1]:
                      weight = duration
                      start_time = shortest_paths[current_node][4]
                      end_time = start_time + pd.Timedelta(minutes= weight)
                      wait_time = 0
                      mode = "walk"
                      trip_id = None
                      service_id = None
                      shortest_paths[neighbor] = (current_node, new_total_time, weight, start_time, end_time, wait_time, duration, mode, trip_id, service_id)
                      heapq.heappush(pq, (new_total_time, neighbor))
  if end_node not in shortest_paths:
        return None, [], [], [], [], []
    
 # prev_node, total_time, weight (wait + actual), start_time, end_time, wait_time, duration, mode, trip id, service_id
  path = []
  modes = []
  start_times = []
  end_times = []
  wait_times = []
  durations = []
  trip_ids = []
  routes = []
  current_node = end_node

  while current_node:
        path.append(current_node)
        prev_node = shortest_paths[current_node]
        prev_node, _, _, start_time, end_time, wait_time, duration, mode, trip_id, service_id = shortest_paths[current_node]
        
       
        current_node = prev_node
        if mode:
            start_times.append(start_time)
            wait_times.append(wait_time)
            end_times.append(end_time)
            durations.append(duration)
            modes.append(mode)
            trip_ids.append(trip_id)
            routes.append(service_id)
        

  path.reverse()
  start_times.reverse()
  end_times.reverse()
  wait_times.reverse()
  durations.reverse()
  modes.reverse()
  trip_ids.reverse()
  routes.reverse()
  
 


  return find_earlier_stop(G, path, start_times, end_times, wait_times, durations, modes, trip_ids, routes, path[0], starting_time)               






def return_instructions(path, start_times, end_times, wait_times, durations, modes, trip_ids, routes):
    return_string = ""
    if not path:
        return_string += ("No path found.")
        return return_string

    print("Instructions:")
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        mode = modes[i]
        duration = durations[i]
        start_time = start_times[i]
        end_time = end_times[i]
        wait_time = wait_times[i]
        trip_id = trip_ids[i]
        route = routes[i]
        
     
        


        if mode == "walk":
            return_string += (f"Walk from {node} to {next_node}, estimated time: {duration:.2f} minutes.")
            return_string += (f" Start time: {start_time.time()}, End time: {end_time.time()}.")
        elif mode == "transit":
            return_string +=(f" Take route {route} from {node} to {next_node} (Trip ID: {trip_id}).")
            if start_time is not None:
                return_string +=(f" Departure time: {start_time.time()}, travel duration: {duration:.2f} minutes.")
                return_string +=(f" Wait time: {wait_time:.2f} minutes, End time: {end_time.time()}.")

    # Print arrival time at the last node
    return_string +=(f" Arrive at {path[1]} at {end_times[-1].time()}.")
    return return_string


#i need to get the user's current location as long and latutide as well as their end node 
def find_shortest_path_with_user(G, start_address, end_address, current_time_str, current_date_str):
  start_node, end_node = add_start_end_to_graph(G, start_address, end_address)
  if start_node and end_node:
      path, start_times, end_times, wait_times, durations, modes, trip_ids, routes = find_shortest_path(G, start_node, end_node, current_time_str, current_date_str)
      return path, start_times, end_times, wait_times, durations, modes, trip_ids, routes
  else:
      return None, [], [], [], []
  
def add_start_end_to_graph(G, start_address, end_address):
  start_lat, start_lon = address_to_latlong(start_address)
  print("success")
  end_lat, end_lon = address_to_latlong(end_address)
  print("success")
  
  if start_lat is not None and end_lat is not None and end_lon is not None and start_lon is not None:
      start_node = 'start_location'
      end_node = 'end_location'
      G.add_node(start_node, stop_lat= start_lat, stop_lon= start_lon)
      G.add_node(end_node, stop_lat= end_lat, stop_lon= end_lon)
      nodes = list(G.nodes(data=True))
      for node, data in nodes:
          if node != start_node:
              walk_time = calculate_walk_time(start_lat, start_lon, data['stop_lat'], data['stop_lon'])
              if walk_time < 30:
                  G.add_edge(start_node, node, mode="walk", weight=walk_time)
                  G.add_edge(node, start_node, mode="walk", weight=walk_time)
          if node != end_node:
              walk_time = calculate_walk_time(end_lat, end_lon, data['stop_lat'], data['stop_lon'])
              if walk_time < 30:
                  G.add_edge(end_node, node, mode="walk", weight=walk_time)
                  G.add_edge(node, end_node, mode="walk", weight=walk_time)
      return start_node, end_node
  
def address_to_latlong(address):
    # Geoapify API key
    api_key = os.getenv('GEOAPIFY_API_KEY')
    
    # Make a request to the Geoapify Geocoding API
    url = f"https://api.geoapify.com/v1/geocode/search?text={address}&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data['features']:
            location = data['features'][0]['geometry']['coordinates']
            return location[1], location[0]  # Return latitude, longitude
        else:
            print("Location could not be geocoded.")
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

        

def return_attributes(path, start_times, end_times, durations, trip_ids, start_address, end_address):
    stops = pd.read_csv('stops.txt')
    stop_names = stops.set_index('stop_id')['stop_name'].to_dict()
    trips_df = pd.read_csv('trips.txt')
    routes_df = pd.read_csv('routes.txt')
    route_ids = trips_df.set_index('trip_id')['route_id'].to_dict()
    route_info = routes_df.set_index('route_id')[['route_short_name', 'route_long_name', 'route_color']].to_dict('index')
    trip_headsigns = trips_df.set_index('trip_id')['trip_headsign'].to_dict()

    # Initialize merged lists
    merged_path = []
    merged_start_times = []
    merged_end_times = []
    merged_durations = []
    merged_trip_ids = []
    
    # Merge continuous walking segments
    i = 0
    while i < len(trip_ids):
        if trip_ids[i] is None:
            walk_start_index = i
            walk_duration = durations[i]
            
            while i + 1 < len(trip_ids) and trip_ids[i + 1] is None:
                walk_duration += durations[i + 1]
                i += 1
            
            merged_path.append(path[walk_start_index])
            merged_start_times.append(start_times[walk_start_index])
            merged_end_times.append(end_times[i])
            merged_durations.append(walk_duration)
            merged_trip_ids.append(None)
        else:
            merged_path.append(path[i])
            merged_start_times.append(start_times[i])
            merged_end_times.append(end_times[i])
            merged_durations.append(durations[i])
            merged_trip_ids.append(trip_ids[i])
        
        i += 1

    # Append the final destination to the path
    merged_path.append(path[-1])

    legs = []
    colors = []
    for trip in merged_trip_ids:
        if trip is None:
            legs.append("Walk")
            colors.append("#707372")
        else:
            route_id = route_ids.get(trip)
            route_name = route_info.get(route_id)['route_long_name']
            route_number = route_info.get(route_id)['route_short_name']
            trip_headsign = trip_headsigns.get(trip)
            route_color = route_info.get(route_id)['route_color']
            legs.append(trip_headsign + " (" + str(route_number) + " " + route_name + ")")
            colors.append("#" + route_color)

    times = merged_start_times + [merged_end_times[-1]]

    # Round up durations to the nearest minute
    rounded_durations = [math.ceil(duration) for duration in merged_durations]

    # Replace placeholders with actual addresses
    merged_path = [start_address if node == "start_location" else end_address if node == "end_location" else stop_names.get(node) for node in merged_path]

    return merged_path, times, rounded_durations, legs, colors

def fetch_real_time_data(stop_id):
    # API key for accessing the CUMTD API
    api_key = os.getenv('CUMTD_API_KEY')
    url = f"https://developer.cumtd.com/api/v2.2/json/GetDeparturesByStop?key={api_key}&stop_id={stop_id}"
    response = requests.get(url)
    data = response.json()
    return data

def update_graph_with_real_time_data(G, start_address, end_address, current_time_str, current_date_str, starting_trip_ids, trip_ids_count, starting_indexes, path, start_times, end_times, durations, trip_ids):
    #iterates through list of unique trip ids 
  #[4,4]
     #[1,5]
        #[0,5]
        #a 1 b 1 c 1 d 1 e None f 5 g 5 h 5 i 5 
        
        #['[@6.0.14328849@][2][1622467715328]/17__I1_NONUI_MF'], [1], [18]
        delay = 0
        
        #has a rnage of 2 
        for i in range(len(starting_trip_ids)):
            #pulls a 
            starting_stop = path[starting_indexes[i]]
            print("Starting stop " + starting_stop)
          
            #gets real_time_date for a 
            real_time_data = fetch_real_time_data(starting_stop)
            if 'departures' in real_time_data and real_time_data['departures']:
                #have to add additonal logic here for when preview is too far ahead 
                
                #gets the deprature from a that has same trip_id 
                matching_departure = next((departure for departure in real_time_data['departures'] if departure['trip']['trip_id'] == starting_trip_ids[i]), None)
                
                if matching_departure:
                    expected_time_str = matching_departure['expected']
                   
                    expected_time = pd.to_datetime(expected_time_str).tz_localize(None).time()  # Make timezone-naive
                    scheduled_time = start_times[starting_indexes[i]].tz_localize(None).time()  # Make timezone-naive
                    print(expected_time)
                    print(scheduled_time)

                    
                    #gets the delay 
                    expected_minutes = expected_time.hour * 60 + expected_time.minute 
                    scheduled_minutes = scheduled_time.hour * 60 + scheduled_time.minute 

    # Calculate the delay in minutes
                    delay = (expected_minutes - scheduled_minutes)
                    print(delay)
                    #if no delay, do not care 
                    if delay <= 0:
                        continue
            #atp we have calculated the delay. Now we need to add that delay to all the edges in the graph that match that trip_id
            
            #this will give 0 
            index = starting_indexes[i]
            
            #this will give 4 
            amount_in_path = trip_ids_count[i]
            
       
            while (amount_in_path > 0):
                stop_id = path[index]
                # a and b 
                # b and c 
                #c and d 
                #d and e 
                next_stop_id = path[index + 1]
                if stop_id != 'start_location' and stop_id != 'end_location' and next_stop_id != 'start_location' and next_stop_id != 'end_location':
                    for key, edge_data in G[stop_id][next_stop_id].items():
                        #finds the edge with the same trip_id 
                        if  edge_data['mode'] == 'transit' and edge_data['trip_id'] == starting_trip_ids[i]:
                            #edits their start and end time 
                            new_start_time = (pd.to_datetime(edge_data['start_time'], format='%H:%M:%S') + pd.Timedelta(minutes=delay)).strftime('%H:%M:%S')
                            new_end_time = (pd.to_datetime(edge_data['end_time'], format='%H:%M:%S') + pd.Timedelta(minutes=delay)).strftime('%H:%M:%S')
                            print(f"  Original start_time: {edge_data['start_time']}")
                            print(f"  Original end_time: {edge_data['end_time']}")
                            print(stop_id)
                            print(next_stop_id)
                            G[stop_id][next_stop_id][key]['start_time'] = new_start_time
                            G[stop_id][next_stop_id][key]['end_time'] = new_end_time
                            print_edges_between_stops(G, stop_id, next_stop_id, starting_trip_ids[i])
                            print(f"New start_time: {G[stop_id][next_stop_id][key]['start_time']}")
                            print(f"New end_time: {G[stop_id][next_stop_id][key]['end_time']}")
                       
                amount_in_path -=1
                #3 
                #2
                #1
                #0 
                index += 1
                #1
                #2
                #3
                #4 
                pq = None
                shortest_paths = None
              
        return find_shortest_path_with_user(G, start_address, end_address, current_time_str, current_date_str)
        
        

def find_shortest_path_with_live_updates(G, path, start_address, end_address, current_time_str, current_date_str, start_times, end_times, durations, trip_ids):
   
    print("fresh start")
    max_iterations = 5  # Set a maximum number of iterations to prevent infinite loops
    iteration = 0
    while iteration < max_iterations: 
        iteration+=1
        global_trip_id = ""
        #a list of unique trip ids
        starting_trip_ids = []
        #how many times that trip_id shows 
        trip_ids_count = []
        #the index in the trip_id list where the first occurence happens
        starting_indexes = []
        
        #check all the trip_ids
        #- 1 - 1 - 1 - 1 - None - 5 - 5 - 5 - 5 
        #8
        
       
        for i in range(len(trip_ids)):
            
           
         
            #if it matches the global trip id, then we have two of that trip_id
            if trip_ids[i] == global_trip_id:
                
                trip_ids_count[-1]+=1
                 #[4,4]
            elif (trip_ids[i] != None):
                
                #we get a new trip_id 
                global_trip_id = trip_ids[i]
                #global becomes 1 
                #global becomes 5 
                #we add it to our list 
                starting_trip_ids.append(global_trip_id)
                #[1,5]
                #we store the index of the trip_id 
                starting_indexes.append(i)
                #[0,5]
                trip_ids_count.append(1)
                #[1,]
                #[4,1]
           
                
                
        
        original_path = path.copy()
        
        
        path, start_times, end_times, wait_times, durations, modes, trip_ids, routes = update_graph_with_real_time_data(G, start_address, end_address, current_time_str, current_date_str, starting_trip_ids, trip_ids_count, starting_indexes, path, start_times, end_times, durations, trip_ids)
        
        print("calculating")
        if path == original_path:
            
            
            return path, start_times, end_times, wait_times, durations, modes, trip_ids, routes
        else:
            print("considering something else")
  
  
     

            
def print_edges_between_stops(G, stop1, stop2, trip_id):
    """
    Prints all the edges between two stops in the graph that have a specific trip_id.
    
    Parameters:
    G (networkx.Graph): The graph object.
    stop1 (str): The ID of the first stop.
    stop2 (str): The ID of the second stop.
    trip_id (str): The trip ID to filter the edges by.
    """
    if G.has_edge(stop1, stop2):
        edges = G.get_edge_data(stop1, stop2)
        print(f"Edges between {stop1} and {stop2} with trip_id {trip_id}:")
        found = False
        for key, edge_data in edges.items():
            if edge_data.get('trip_id') == trip_id:
                print(f"  Edge {key}: {edge_data}")
                found = True
        if not found:
            print(f"No edges found with trip_id {trip_id}.")
    else:
        print(f"No edges found between {stop1} and {stop2}.")
        
def call_print_edges_between_stops(G, path, trip_id):
    """
    Calls print_edges_between_stops for each consecutive pair of stops in the path.
    
    Parameters:
    G (networkx.Graph): The graph object.
    path (list): The list of stops in the path.
    trip_id (str): The trip ID to filter the edges by.
    """
    for i in range(len(path) - 1):
        stop1 = path[i]
        stop2 = path[i + 1]
        print_edges_between_stops(G, stop1, stop2, trip_id)
        



