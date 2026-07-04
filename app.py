import os
import csv
import json
from datetime import datetime
from io import StringIO
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, make_response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash

# Import DB and ML Modules
from database.db_manager import (
    init_db, add_user, get_user_by_id, get_user_by_username,
    save_prediction, save_suitability_assessment, log_action,
    get_all_predictions, get_user_predictions, get_all_users,
    delete_user, get_system_logs, get_system_analytics_summary
)
from models.preprocessing import validate_inputs
from models.prediction import predict_crop_recommendation, assess_crop_suitability, CROP_METADATA
from models.train_model import run_training_pipeline, METRICS_PATH

# FPDF2 for PDF Generation
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "opticrops_super_secret_session_key_98231"

# Initialize Login Manager
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user_id, username, role):
        self.id = user_id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    row = get_user_by_id(user_id)
    if row:
        return User(row['id'], row['username'], row['role'])
    return None

# Context Processor to inject crop list to all templates
@app.context_processor
def inject_crop_list():
    return dict(crop_list=CROP_METADATA)

# --- PDF Report Class ---
class AnalyticalPDFReport(FPDF):
    def header(self):
        # Draw dark green banner at top
        self.set_fill_color(27, 67, 50)
        self.rect(0, 0, 210, 38, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 18)
        self.cell(0, 10, 'OPTICROP - ANALYTICAL REPORT', ln=True, align='C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, 'Smart Agricultural Production Optimization Engine', ln=True, align='C')
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user_row = get_user_by_username(username)
        if user_row and check_password_hash(user_row["password_hash"], password):
            user_obj = User(user_row["id"], user_row["username"], user_row["role"])
            login_user(user_obj)
            log_action("User Login", user_row["id"], f"Logged in from web portal.")
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        username = request.form.get("username")
        role = request.form.get("role")
        password = request.form.get("password")
        
        if not username or not role or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")
            
        success, message = add_user(username, password, role)
        if success:
            flash("Account registered successfully! Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash(message, "error")
            
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    log_action("User Logout", current_user.id, f"Logged out successfully.")
    logout_user()
    flash("You have logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    summary = get_system_analytics_summary()
    
    # Get last 5 predictions
    if current_user.role == 'Admin':
        recent_preds = get_all_predictions()[:5]
    else:
        recent_preds = get_user_predictions(current_user.id)[:5]
        
    today_str = datetime.now().strftime("%B %d, %Y")
    
    return render_template("dashboard.html", summary=summary, recent_predictions=recent_preds, today_date=today_str, active_page='dashboard')

@app.route("/recommendation", methods=["GET", "POST"])
@login_required
def recommendation():
    result = None
    inputs = None
    if request.method == "POST":
        try:
            n = float(request.form.get("N"))
            p = float(request.form.get("P"))
            k = float(request.form.get("K"))
            temp = float(request.form.get("temperature"))
            hum = float(request.form.get("humidity"))
            ph = float(request.form.get("ph"))
            rain = float(request.form.get("rainfall"))
        except (ValueError, TypeError):
            flash("Invalid input values. Please check all fields.", "error")
            return render_template("recommendation.html", active_page='recommendation')

        inputs = {'N': n, 'P': p, 'K': k, 'temperature': temp, 'humidity': hum, 'ph': ph, 'rainfall': rain}
        
        is_valid, err = validate_inputs(n, p, k, temp, hum, ph, rain)
        if not is_valid:
            flash(err, "error")
            return render_template("recommendation.html", inputs=inputs, active_page='recommendation')
            
        # Call Prediction Engine
        pred_res = predict_crop_recommendation(n, p, k, temp, hum, ph, rain)
        if 'error' in pred_res:
            flash(pred_res['error'], "error")
        else:
            result = pred_res
            # Save Prediction to Database
            save_prediction(
                current_user.id, n, p, k, temp, hum, ph, rain,
                result['crop_label'], result['confidence_score'],
                result['yield_potential'], result['season']
            )
            log_action("Crop Recommendation", current_user.id, f"Recommended crop: {result['crop_name']} with confidence {result['confidence_score']}%")

    return render_template("recommendation.html", result=result, inputs=inputs, active_page='recommendation')

@app.route("/suitability", methods=["GET", "POST"])
@login_required
def suitability():
    result = None
    inputs = None
    if request.method == "POST":
        crop = request.form.get("crop")
        try:
            n = float(request.form.get("N"))
            p = float(request.form.get("P"))
            k = float(request.form.get("K"))
            temp = float(request.form.get("temperature"))
            hum = float(request.form.get("humidity"))
            ph = float(request.form.get("ph"))
            rain = float(request.form.get("rainfall"))
        except (ValueError, TypeError):
            flash("Invalid input values. Please check all fields.", "error")
            return render_template("suitability.html", active_page='suitability')

        inputs = {'crop': crop, 'N': n, 'P': p, 'K': k, 'temperature': temp, 'humidity': hum, 'ph': ph, 'rainfall': rain}
        
        is_valid, err = validate_inputs(n, p, k, temp, hum, ph, rain)
        if not is_valid:
            flash(err, "error")
            return render_template("suitability.html", inputs=inputs, active_page='suitability')
            
        # Evaluate Suitability
        result = assess_crop_suitability(crop, n, p, k, temp, hum, ph, rain)
        
        # Save suitability assessment
        save_suitability_assessment(
            current_user.id, crop, n, p, k, temp, hum, ph, rain,
            result['compatibility_score'], result['suitability_rating']
        )
        log_action("Suitability Assessment", current_user.id, f"Assessed suitability for crop: {crop} | Score: {result['compatibility_score']}% ({result['suitability_rating']})")

    return render_template("suitability.html", result=result, inputs=inputs, active_page='suitability')

@app.route("/analytics")
@login_required
def analytics():
    # Only Researchers and Admins can view analytics
    if current_user.role not in ["Researcher", "Admin"]:
        flash("Unauthorized. Research analytics are restricted to Researchers and Admins.", "error")
        return redirect(url_for("dashboard"))
        
    return render_template("analytics.html", active_page='analytics')

@app.route("/reports")
@login_required
def reports():
    search_query = request.args.get("search")
    crop_filter = request.args.get("crop_filter")
    
    if current_user.role in ['Admin', 'Researcher']:
        predictions = get_all_predictions(search_query, crop_filter)
    else:
        predictions = get_user_predictions(current_user.id, search_query)
        
    return render_template("reports.html", predictions=predictions, search_query=search_query, crop_filter=crop_filter, active_page='reports')

@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if current_user.role != 'Admin':
        flash("Access Denied. Admin Dashboard is restricted to Administrators.", "error")
        return redirect(url_for("dashboard"))
        
    users = get_all_users()
    logs = get_system_logs()
    
    # Load model performance metrics if trained
    metrics = None
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass
            
    return render_template("admin_dashboard.html", users=users, logs=logs, metrics=metrics, active_page='admin_dashboard')

@app.route("/admin/create_user", methods=["POST"])
@login_required
def admin_create_user():
    if current_user.role != 'Admin':
        flash("Unauthorized.", "error")
        return redirect(url_for("dashboard"))
        
    username = request.form.get("username")
    role = request.form.get("role")
    password = request.form.get("password")
    
    success, message = add_user(username, password, role)
    if success:
        flash(f"User account '{username}' (Role: {role}) created successfully.", "success")
    else:
        flash(message, "error")
        
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@login_required
def admin_delete_user_route(user_id):
    if current_user.role != 'Admin':
        flash("Unauthorized.", "error")
        return redirect(url_for("dashboard"))
        
    success, message = delete_user(user_id, current_user.id)
    if success:
        flash(message, "success")
    else:
        flash(message, "error")
        
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/retrain", methods=["POST"])
@login_required
def retrain_model_route():
    if current_user.role != 'Admin':
        flash("Access Denied. Retraining requires Admin privileges.", "error")
        return redirect(url_for("dashboard"))
        
    try:
        metrics = run_training_pipeline()
        log_action("Model Retraining", current_user.id, f"Triggered model retraining. Champion: {metrics['best_model']} (Accuracy: {metrics['accuracy']:.4f})")
        flash(f"Models successfully retrained! The new champion model is {metrics['best_model']} with a validation accuracy of {metrics['accuracy']*100:.2f}%.", "success")
    except Exception as e:
        log_action("Model Retraining Failure", current_user.id, f"Error: {str(e)}")
        flash(f"Retraining failed: {str(e)}", "error")
        
    return redirect(url_for("admin_dashboard"))

# --- DOWNLOAD & EXPORTS ---

@app.route("/exports/csv")
@login_required
def download_predictions_csv():
    # Fetch historical logs
    if current_user.role in ['Admin', 'Researcher']:
        preds = get_all_predictions()
    else:
        preds = get_user_predictions(current_user.id)
        
    # Generate CSV in memory
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Prediction_ID', 'Timestamp', 'User_ID', 'Username', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Predicted_Crop', 'Confidence_Score', 'Yield_Potential', 'Sowing_Season'])
    
    for p in preds:
        cw.writerow([
            p['id'], p['created_at'], p['user_id'], 
            p['username'] if 'username' in p.keys() else current_user.username,
            p['N'], p['P'], p['K'], p['temperature'], p['humidity'], 
            p['ph'], p['rainfall'], p['predicted_crop'], 
            p['confidence_score'], p['yield_potential'], p['sowing_season']
        ])
        
    response = make_response(si.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=opticrops_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route("/exports/pdf")
@login_required
def download_pdf_report():
    if current_user.role not in ['Admin', 'Researcher']:
        flash("Unauthorized. PDF reports require Researcher or Admin credentials.", "error")
        return redirect(url_for("dashboard"))
        
    summary = get_system_analytics_summary()
    predictions = get_all_predictions()[:10]  # Show top 10 logs in PDF
    
    pdf = AnalyticalPDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(33, 33, 33)
    
    # 1. Executive Summary
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(27, 67, 50)
    pdf.cell(0, 10, '1. Executive Summary & Core Analytics', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    summary_text = (
        f"This report summarizes precision agriculture data processed by the OptiCrop decision-support system. "
        f"To date, the system has logged a total of {summary['total_predictions']} crop predictions requested by system users. "
        f"The most recommended crop based on overall climate compatibility runs is '{summary['most_recommended_crop'].capitalize()}'. "
        f"Average rainfall recorded stands at {summary['average_rainfall']} mm, with average soil nutrient metrics at "
        f"Nitrogen: {summary['soil_stats']['avg_N']} kg/ha, Phosphorus: {summary['soil_stats']['avg_P']} kg/ha, and Potassium: {summary['soil_stats']['avg_K']} kg/ha."
    )
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(5)
    
    # 2. Performance Table
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(27, 67, 50)
    pdf.cell(0, 10, '2. Diagnostic Prediction Log (Recent Runs)', ln=True)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(216, 243, 220)
    pdf.set_text_color(33, 33, 33)
    
    # Table headers
    pdf.cell(30, 8, 'Timestamp', border=1, fill=True)
    pdf.cell(20, 8, 'User', border=1, fill=True)
    pdf.cell(35, 8, 'N-P-K (kg/ha)', border=1, align='C', fill=True)
    pdf.cell(20, 8, 'pH', border=1, align='C', fill=True)
    pdf.cell(30, 8, 'Crop Recommendation', border=1, fill=True)
    pdf.cell(20, 8, 'Confidence', border=1, align='C', fill=True)
    pdf.cell(30, 8, 'Sowing Season', border=1, fill=True)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    for p in predictions:
        ts = p['created_at'][:16]
        user = p['username'] if p['username'] else 'Unknown'
        npk = f"{int(p['N'])}-{int(p['P'])}-{int(p['K'])}"
        ph = f"{p['ph']:.1f}"
        crop = p['predicted_crop'].capitalize()
        conf = f"{p['confidence_score']:.1f}%"
        season = p['sowing_season'].split(" (")[0]
        
        pdf.cell(30, 7, ts, border=1)
        pdf.cell(20, 7, user[:10], border=1)
        pdf.cell(35, 7, npk, border=1, align='C')
        pdf.cell(20, 7, ph, border=1, align='C')
        pdf.cell(30, 7, crop, border=1)
        pdf.cell(20, 7, conf, border=1, align='C')
        pdf.cell(30, 7, season[:18], border=1)
        pdf.ln()
        
    # Page break for charts
    pdf.add_page()
    
    # 3. Model Accuracy Comparison Plot
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(27, 67, 50)
    pdf.cell(0, 10, '3. Machine Learning Calibration & Correlation Plots', ln=True)
    pdf.ln(2)
    
    # Embed correlation heatmap if exists
    heatmap_path = "static/charts/correlation_heatmap.png"
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=10, y=45, w=90)
        
    # Embed model comparison if exists
    comparison_path = "static/charts/model_comparison.png"
    if os.path.exists(comparison_path):
        pdf.image(comparison_path, x=105, y=45, w=95)
        
    pdf.set_y(120)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, 'Figure 1: Pearson Correlation Matrix (Left) & Algorithm Validation Benchmark (Right)', ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(27, 67, 50)
    pdf.cell(0, 8, 'Interpretation Guidelines:', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    interpretation_text = (
        "- Soil Nutrients: Heavy N-P-K correlations dictate nutrient lockups. For instance, high Potassium (K) "
        "strongly correlates with specific temperate fruits (Apple, Grapes) that exhibit high potassium withdrawal indices.\n"
        "- Model Selection: The validation engine evaluates Decision Trees, SVM, Naive Bayes, KNN, and Random Forests. "
        "A Stratified K-Fold validation prevents overfitting, choosing the estimator with minimal validation drift."
    )
    pdf.multi_cell(0, 5, interpretation_text)
    
    # Return PDF response
    pdf_bytes = pdf.output(dest='S')
    response = make_response(pdf_bytes)
    response.headers["Content-Disposition"] = f"attachment; filename=opticrops_research_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    response.headers["Content-type"] = "application/pdf"
    return response

# --- APP BOOTSTRAP ---

if __name__ == "__main__":
    print("OptiCrop web application startup...")
    # Initialize Database Schema & default credentials
    init_db()
    
    # Start Flask Web Server (Runs locally on localhost:5000)
    app.run(host="127.0.0.1", port=5000, debug=True)
