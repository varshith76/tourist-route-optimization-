
from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random value

# ---------- Admin Login Route ----------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username != 'admin':
            error = 'Only admin can log in here.'
        else:
            conn = sqlite3.connect('users.db')
            cur = conn.cursor()
            cur.execute('SELECT password FROM users WHERE username = ?', (username,))
            row = cur.fetchone()
            conn.close()
            if row and check_password_hash(row[0], password):
                session['username'] = username
                return redirect(url_for('admin_users'))
            else:
                error = 'Invalid admin credentials.'
    return render_template('admin_login.html', error=error)

# ---------- Admin: View Users Route ----------
@app.route('/admin/users')
def admin_users():
    if 'username' not in session or session['username'] != 'admin':
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute('SELECT id, username FROM users')
    users = cur.fetchall()
    conn.close()
    return render_template('admin_users.html', users=users)


# ---------- Flask route ----------


# ---------- User Registration Route ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            error = 'Please provide both username and password.'
        else:
            conn = sqlite3.connect('users.db')
            cur = conn.cursor()
            try:
                cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, generate_password_hash(password)))
                conn.commit()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()
    return render_template('register.html', error=error, success=success)

# ---------- User Logout Route ----------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ---------- User Login Route ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin':
            error = 'Admin must log in at /admin/login.'
        else:
            conn = sqlite3.connect('users.db')
            cur = conn.cursor()
            cur.execute('SELECT password FROM users WHERE username = ?', (username,))
            row = cur.fetchone()
            conn.close()
            if row and check_password_hash(row[0], password):
                session['username'] = username
                return redirect(url_for('index'))
            else:
                error = 'Invalid username or password.'
    return render_template('login.html', error=error)

# ---------- Load dataset ----------
df = pd.read_csv("combined_telangana_places.csv")
# Columns: Place, Latitude, Longitude, Description, Category

# ---------- Load ML model and encoder ----------
model = joblib.load("combined_places_model.pkl")
le = joblib.load("category_encoder.pkl")

# Ensure new categories (e.g., EV charging stations) don't crash prediction.
# LabelEncoder throws if it sees unseen labels; we append any new categories here.
for extra_cat in ["Public Charging"]:
    if extra_cat not in le.classes_:
        le.classes_ = np.append(le.classes_, extra_cat)

# ---------- Haversine function ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------- Generate features ----------
def generate_features(df, start_coords, dest_coords, max_distance=None):
    """Compute distance features for all rows. If max_distance is provided, return only rows within that distance
    from start or dest as well (keeps backward compatibility). Returns (features_df, indices_list).
    """
    features = []
    filtered_indices = []
    for i, row in df.iterrows():
        dist_start = haversine(start_coords[0], start_coords[1], row['Latitude'], row['Longitude'])
        dist_dest = haversine(dest_coords[0], dest_coords[1], row['Latitude'], row['Longitude'])
        cat_enc = le.transform([row["Category"]])[0]
        features.append({
            "Distance_Start": dist_start,
            "Distance_Dest": dist_dest,
            "Category_Encoded": cat_enc
        })
        # If a max_distance is provided, record only indices that satisfy it. If not provided, record all indices.
        if max_distance is None or dist_start <= max_distance or dist_dest <= max_distance:
            filtered_indices.append(i)
    return pd.DataFrame(features), filtered_indices

# ---------- Main Route ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    top_places = None
    location_names = list(df["Place"].unique())  # For dropdown
    error_msg = None

    if request.method == "POST":
        start_name = request.form["start_name"]
        dest_name = request.form["dest_name"]

        # Lookup coordinates
        try:
            start_row = df[df["Place"].str.lower() == start_name.lower()].iloc[0]
            dest_row = df[df["Place"].str.lower() == dest_name.lower()].iloc[0]
        except IndexError:
            error_msg = "Invalid start or destination name."
            return render_template("index.html", top_places=None, error=error_msg, locations=location_names, username=session.get('username'))

        start_coords = (start_row["Latitude"], start_row["Longitude"])
        dest_coords = (dest_row["Latitude"], dest_row["Longitude"])

        # Generate features for all rows (do not prefilter by max distance so on-route places aren't excluded)
        features_all, filtered_indices = generate_features(df, start_coords, dest_coords, max_distance=None)

        # Keep only the feature rows that correspond to filtered_indices so lengths match
        features = features_all.iloc[filtered_indices].reset_index(drop=True)

        if features.empty:
            error_msg = "No places found near the route."
            return render_template("index.html", top_places=None, error=error_msg, locations=location_names, username=session.get('username'))

        df_filtered = df.iloc[filtered_indices].reset_index(drop=True).copy()
        df_filtered["Predicted_Score"] = model.predict(features)
        # Add distance columns (km) and a minimum distance to either start or dest
        # Round to 2 decimal places for display
        df_filtered["Distance_Start_km"] = [round(x, 2) for x in features["Distance_Start"]]
        df_filtered["Distance_Dest_km"] = [round(x, 2) for x in features["Distance_Dest"]]
        df_filtered["Distance_km"] = df_filtered[["Distance_Start_km", "Distance_Dest_km"]].min(axis=1)

        # Compute whether a place lies between start and dest using a corridor projection method.
        # Convert lat/lon differences to x/y (km) using an equirectangular approximation relative to the start point.
        d_total = haversine(start_coords[0], start_coords[1], dest_coords[0], dest_coords[1])
        R = 6371.0
        lat0 = start_coords[0]
        lon0 = start_coords[1]

        lats = df_filtered['Latitude'].astype(float).to_numpy()
        lons = df_filtered['Longitude'].astype(float).to_numpy()

        # Per-point cos(lat) for more accurate x distances
        cos_lat = np.cos(np.radians((lats + lat0) / 2.0))
        dx = R * np.radians(lons - lon0) * cos_lat
        dy = R * np.radians(lats - lat0)

        # Route vector (from start to dest) in same projection
        rx = R * np.radians(dest_coords[1] - lon0) * np.cos(np.radians((dest_coords[0] + lat0) / 2.0))
        ry = R * np.radians(dest_coords[0] - lat0)

        denom = rx * rx + ry * ry
        if denom == 0:
            # start == dest: treat near-start points as along-route
            perp_dist = np.sqrt(dx * dx + dy * dy)
            perp_tol = max(5.0, 0.03 * d_total)
            along_mask = perp_dist <= perp_tol
        else:
            t = (dx * rx + dy * ry) / denom
            # perpendicular distance from point to route line
            perp_dist = np.abs(-ry * dx + rx * dy) / np.sqrt(denom)
            # corridor tolerance: at least 5 km or 3% of route length
            # Use a wider corridor (15 km) to include places that are reasonably near the route.
            perp_tol = 15.0
            along_mask = (t >= 0.0) & (t <= 1.0) & (perp_dist <= perp_tol)

        # Add perpendicular distance column to dataframe for sorting
        df_filtered["Perp_Distance"] = perp_dist

        # Filter by category (Tourist Place, Hotel, Restaurant, Petrol Station)
        # For the 'Tourist Places' section: show all places that are on the way (sorted by distance to route)
        top_tourist = df_filtered[df_filtered["Category"].isin(["Historical","Lake","Entertainment","Dam","Religious","Nature","Wildlife","Cultural"])].sort_values(by="Perp_Distance", ascending=True).head(10)
        top_hotels = df_filtered[df_filtered["Category"]=="Hotel"].sort_values(by="Perp_Distance", ascending=True).head(10)
        top_restaurants = df_filtered[df_filtered["Category"]=="Restaurant"].sort_values(by="Perp_Distance", ascending=True).head(10)
        top_petrol = df_filtered[df_filtered["Category"]=="Petrol Station"].sort_values(by="Perp_Distance", ascending=True).head(10)
        top_ev_chargers = df_filtered[df_filtered["Category"]=="Public Charging"].sort_values(by="Perp_Distance", ascending=True).head(10)

        top_places = {
            "Tourist Places": top_tourist,
            "Hotels": top_hotels,
            "Restaurants": top_restaurants,
            "Petrol Stations": top_petrol,
            "EV Charging Stations": top_ev_chargers
        }
    
    # Pass start/dest coords so template can build map links or interactive map
    start_coords = None
    dest_coords = None
    if request.method == "POST" and top_places is not None:
        start_coords = (start_row["Latitude"], start_row["Longitude"]) if 'start_row' in locals() else None
        dest_coords = (dest_row["Latitude"], dest_row["Longitude"]) if 'dest_row' in locals() else None

    return render_template("index.html", top_places=top_places, error=error_msg, locations=location_names, start_coords=start_coords, dest_coords=dest_coords)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
