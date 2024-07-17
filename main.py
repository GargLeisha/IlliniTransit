from flask import Flask, render_template, request, make_response
from routes import find_shortest_path_with_user, initialize_graph, return_attributes, find_shortest_path_with_live_updates, return_instructions, calculate_walk_time, address_to_latlong, return_attributes
from datetime import datetime
import pytz
from urllib.parse import quote
from waitress import serve
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

@app.route('/')
@app.route('/index')

def index():
    google_api_key = os.getenv('MAP_KEY')
    geoapify_api_key = os.getenv('GEOAPIFY_API_KEY')
    return render_template("index.html", google_api_key=google_api_key, geoapify_api_key=geoapify_api_key)

@app.route('/transit1')
def get_route():
    try:
        G = initialize_graph()
        start_point = request.args.get('start_point')
        end_point = request.args.get('end_point')
        central_tz = pytz.timezone('US/Central')
        central_time = datetime.now(central_tz)
        current_time = central_time.strftime("%H:%M:%S")
        current_date = central_time.strftime("%Y%m%d")
        path, start_times, end_times, wait_times, durations, modes, trip_ids, routes = find_shortest_path_with_user(G, start_point, end_point, current_time, current_date)
        path, start_times, end_times, wait_times, durations, modes, trip_ids, routes = find_shortest_path_with_live_updates(G, path, start_point, end_point, current_time, current_date, start_times, end_times, durations, trip_ids)
        path, times, durations, legs, colors = return_attributes(path, start_times, end_times, durations, trip_ids, start_point, end_point)

        # Combine the attributes into a list of dictionaries
        timeline_data = []
        for i in range(len(path)):
            entry = {
                'time': times[i],
                'path': path[i],
                'color': colors[i] if i < len(colors) else "#ddd"  
            }
            if i < len(path) - 1:
                entry['duration'] = durations[i]
                entry['leg'] = legs[i]
            timeline_data.append(entry)

        google_api_key = os.getenv('MAP_KEY')

        if len(path) > 2:
            # Properly encode each waypoint
            waypoints = '|'.join(quote(wp) for wp in path[1:-1])  # Ensure each waypoint is URL encoded
            map_url = f"https://www.google.com/maps/embed/v1/directions?key={google_api_key}&origin={quote(start_point)}&destination={quote(end_point)}&waypoints={waypoints}&mode=transit"
        else:
            map_url = f"https://www.google.com/maps/embed/v1/directions?key={google_api_key}&origin={quote(start_point)}&destination={quote(end_point)}&mode=transit"
        
        response = make_response(render_template("transit1.html", timeline_data=timeline_data, map_url=map_url, google_api_key=os.getenv('MAP_KEY'), geoapify_api_key=os.getenv('GEOAPIFY_API_KEY')))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'
        response.headers['Pragma'] = 'no-cache'

        return response

    except Exception as e:
        print(f"Error processing route: {e}")
        return render_template("500.html"), 500

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5500)
