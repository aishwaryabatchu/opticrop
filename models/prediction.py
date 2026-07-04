import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
DATASET_PATH = os.path.join("dataset", "Crop_recommendation.csv")

# Comprehensive database of crop characteristics
CROP_METADATA = {
    'rice': {
        'name': 'Rice',
        'season': 'Kharif (June - October)',
        'fertilizer': 'Apply NPK in 120:60:40 kg/ha ratio. Split Nitrogen into 3 applications: 50% at planting, 25% at tillering, and 25% at panicle initiation.',
        'irrigation': 'Maintain standing water of 2-5 cm during vegetative and reproductive stages. Drain the field 10-15 days before harvesting.',
        'desc': 'Rice is a staple tropical crop requiring abundant water, high humidity, and prolonged sunshine to mature.',
        'max_yield': 6.2 # tons/ha
    },
    'maize': {
        'name': 'Maize',
        'season': 'Kharif / Spring (June - September / Feb - May)',
        'fertilizer': 'Apply NPK in 100:50:35 kg/ha ratio. Apply zinc sulfate if soil is deficient.',
        'irrigation': 'Provide moderate irrigation. Critical stages for watering are flowering (tasseling/silking) and grain filling.',
        'desc': 'Maize (corn) is a highly versatile cereal crop requiring well-drained loamy soils and warm, sunny climates.',
        'max_yield': 5.5
    },
    'chickpea': {
        'name': 'Chickpea',
        'season': 'Rabi (October - March)',
        'fertilizer': 'Apply NPK in 20:50:20 kg/ha ratio. Being a legume, it fixes nitrogen, so minimal nitrogenous fertilizer is needed.',
        'irrigation': 'Low water requirement. Apply one irrigation at branching and a second at pod development stage.',
        'desc': 'Chickpea is a valuable pulse crop grown in cool, dry seasons. Prefers well-aerated soils to prevent wilt.',
        'max_yield': 2.8
    },
    'kidneybeans': {
        'name': 'Kidney Beans (Rajma)',
        'season': 'Rabi (October - February)',
        'fertilizer': 'Apply NPK in 40:60:20 kg/ha. Needs moderate nitrogen as it fixes less nitrogen compared to other legumes.',
        'irrigation': 'Requires frequent light irrigations. Sensitive to waterlogging and drought; maintain consistent soil moisture.',
        'desc': 'Kidney beans are protein-rich legumes that grow best in mild temperatures and fertile, well-drained soils.',
        'max_yield': 3.0
    },
    'pigeonpeas': {
        'name': 'Pigeon Peas (Arhar/Tur)',
        'season': 'Kharif (June - December)',
        'fertilizer': 'Apply NPK in 25:50:25 kg/ha. Use starter dose of Nitrogen; rely on bacterial nodulation for root nitrogen fixing.',
        'irrigation': 'Drought tolerant. Water during critical stages: bud initiation and pod development if rainfall is insufficient.',
        'desc': 'Pigeon pea is a multi-purpose shrub legume yielding highly nutritious pulses. Prefers warm climates and deep loamy soils.',
        'max_yield': 2.5
    },
    'mothbeans': {
        'name': 'Moth Beans',
        'season': 'Kharif (July - October)',
        'fertilizer': 'Apply NPK in 15:40:10 kg/ha. Requires low input due to excellent adaptability to poor soils.',
        'irrigation': 'Highly drought-resistant. Grows well under dryland farming conditions; minimal irrigation required.',
        'desc': 'Moth bean is an exceptionally drought-tolerant legume, mainly cultivated in arid regions. Serves as excellent soil cover.',
        'max_yield': 1.2
    },
    'mungbean': {
        'name': 'Mungbean (Green Gram)',
        'season': 'Summer / Kharif (March - June / July - October)',
        'fertilizer': 'Apply NPK in 20:40:20 kg/ha. Apply sulfur for improved seed quality and oil content.',
        'irrigation': 'Water every 10-15 days during summer. Critical stages: flowering and pod formation.',
        'desc': 'Mungbean is a short-duration pulse crop that fits well in crop rotation systems. Prefers warm, moist climates.',
        'max_yield': 1.8
    },
    'blackgram': {
        'name': 'Black Gram (Urad)',
        'season': 'Kharif / Summer (June - September / March - June)',
        'fertilizer': 'Apply NPK in 20:40:20 kg/ha. Apply phosphobacter biofertilizer to enhance phosphorus uptake.',
        'irrigation': 'Requires moderate moisture. Irrigate at pod initiation and grain development stages.',
        'desc': 'Black gram is an important pulse crop offering rich proteins and phosphoric acid. Thrives in hot, humid weather.',
        'max_yield': 1.9
    },
    'lentil': {
        'name': 'Lentil',
        'season': 'Rabi (October - March)',
        'fertilizer': 'Apply NPK in 20:40:20 kg/ha. Being a legume, it fixes nitrogen, needing only a starter dose.',
        'irrigation': 'Grown mostly as rainfed. If needed, provide one light irrigation at vegetative stage (35 days) and one at pod formation.',
        'desc': 'Lentil is a nutritious cool-season legume, highly adapted to cold climates and varying soil types.',
        'max_yield': 2.2
    },
    'pomegranate': {
        'name': 'Pomegranate',
        'season': 'Year-round (Best planting in June-July or Sept-Oct)',
        'fertilizer': 'Apply NPK in 600:250:250 g/plant/year for mature trees, along with plentiful organic compost.',
        'irrigation': 'Drip irrigation is highly recommended. Control watering during fruit ripening to prevent cracking.',
        'desc': 'Pomegranate is a hardy, drought-tolerant fruit shrub. Requires hot, dry summers and cool winters for sweet fruit development.',
        'max_yield': 15.0
    },
    'banana': {
        'name': 'Banana',
        'season': 'Year-round (Best planting in June-July)',
        'fertilizer': 'Heavy feeder. Apply NPK in 200:100:300 g/plant. Split Nitrogen and Potash into 4-6 doses.',
        'irrigation': 'Requires high soil moisture. Provide light, frequent waterings every 3-4 days in summer, and 7-8 days in winter.',
        'desc': 'Banana is a fast-growing, heavy-feeding tropical herbaceous plant that thrives in humid, rich, alluvial soils.',
        'max_yield': 35.0
    },
    'mango': {
        'name': 'Mango',
        'season': 'Spring / Summer (Planting in July - August)',
        'fertilizer': 'Apply NPK in 750:200:500 g/tree/year for trees older than 10 years. Fertilize after harvest.',
        'irrigation': 'Water young trees regularly. For bearing trees, stop irrigation 2-3 months before flowering to induce flower buds.',
        'desc': 'Mango is the king of fruits, thriving in tropical and subtropical regions. Requires dry weather during flowering and fruiting.',
        'max_yield': 12.0
    },
    'grapes': {
        'name': 'Grapes',
        'season': 'Winter / Spring (Planting in Jan - Feb)',
        'fertilizer': 'Apply NPK in 500:500:1000 g/vine/year. Potash is essential for sugar synthesis and berry quality.',
        'irrigation': 'Drip irrigation is ideal. Avoid watering during berry ripening to increase sugar concentration and prevent disease.',
        'desc': 'Grapes are woody vines requiring precise training, pruning, and low humidity during fruiting to prevent mildew.',
        'max_yield': 22.0
    },
    'watermelon': {
        'name': 'Watermelon',
        'season': 'Summer (December - March)',
        'fertilizer': 'Apply NPK in 80:50:50 kg/ha. Apply organic manure generously at soil preparation.',
        'irrigation': 'Frequent irrigation is required. Reduce watering as fruit approaches maturity to concentrate sugars.',
        'desc': 'Watermelon is a warm-season vining crop requiring sandy loam soils, high heat, and abundant sunshine.',
        'max_yield': 28.0
    },
    'muskmelon': {
        'name': 'Muskmelon',
        'season': 'Summer (December - March)',
        'fertilizer': 'Apply NPK in 80:40:40 kg/ha. Apply micronutrients like Boron for sweeter fruit.',
        'irrigation': 'Maintain moderate irrigation. Prevent waterlogging to avoid vine rot. Dry conditions are best for sugar retention.',
        'desc': 'Muskmelon is a sweet, aromatic melon variety that thrives in hot climates, dry atmosphere, and sandy river beds.',
        'max_yield': 18.0
    },
    'apple': {
        'name': 'Apple',
        'season': 'Winter (Planting in Jan - March)',
        'fertilizer': 'Apply NPK in 700:350:700 g/tree/year for mature trees. Calcium sprays help prevent bitter pit.',
        'irrigation': 'Requires consistent moisture throughout the growing season, especially during fruit set and development.',
        'desc': 'Apple is a temperate fruit crop requiring cold chilling hours in winter, mild summers, and well-drained soils.',
        'max_yield': 14.0
    },
    'orange': {
        'name': 'Orange',
        'season': 'Year-round (Best planting in June - August)',
        'fertilizer': 'Apply NPK in 600:200:400 g/tree/year. Feed zinc, iron, and manganese foliar sprays if yellowing occurs.',
        'irrigation': 'Moderate irrigation. Irrigate at critical growth phases: flush, flowering, and fruit development.',
        'desc': 'Orange is a popular citrus fruit. Thrives in subtropical climates with warm temperatures and medium rainfall.',
        'max_yield': 16.0
    },
    'papaya': {
        'name': 'Papaya',
        'season': 'Year-round (Planting in Feb-March or July-Sept)',
        'fertilizer': 'Apply NPK in 250:250:500 g/plant/year in split doses. Extremely responsive to organic manure.',
        'irrigation': 'Water every 7-10 days in winter and 5-6 days in summer. Very sensitive to standing water (causes root rot).',
        'desc': 'Papaya is a fast-growing, short-lived herbaceous fruit plant requiring high temperatures and well-drained soils.',
        'max_yield': 40.0
    },
    'coconut': {
        'name': 'Coconut',
        'season': 'Year-round (Best planting in June - September)',
        'fertilizer': 'Apply NPK in 500:320:1200 g/palm/year. Adding common salt (NaCl) improves yield and disease resistance.',
        'irrigation': 'Water generously, especially during summer. Drip irrigation or basin irrigation is highly beneficial.',
        'desc': 'Coconut palm is a coastal tropical crop thriving in sandy, saline soils and high rainfall/relative humidity.',
        'max_yield': 80.0 # nuts/palm/year
    },
    'cotton': {
        'name': 'Cotton',
        'season': 'Kharif (April - June)',
        'fertilizer': 'Apply NPK in 100:50:50 kg/ha. Needs magnesium and boron for optimal boll retention and fiber quality.',
        'irrigation': 'Irrigate at critical stages: squaring, flowering, and boll development. Avoid excess water during early stages.',
        'desc': 'Cotton is a cash crop producing soft fiber. Requires a long frost-free period, high heat, and low rainfall during harvest.',
        'max_yield': 2.4
    },
    'jute': {
        'name': 'Jute',
        'season': 'Kharif (March - May)',
        'fertilizer': 'Apply NPK in 60:30:30 kg/ha. Nitrogen is key for fiber length and quality; split into two doses.',
        'irrigation': 'Requires high rainfall and soil moisture. If dry, provide weekly irrigation during early growth.',
        'desc': 'Jute (golden fiber) is a fast-growing herbaceous plant cultivated for its strong bast fiber. Needs warm, wet conditions.',
        'max_yield': 3.2
    },
    'coffee': {
        'name': 'Coffee',
        'season': 'Year-round (Planting in June - August)',
        'fertilizer': 'Apply NPK in 140:90:120 kg/ha. Apply organic compost and mulch to maintain humus layer.',
        'irrigation': 'Requires wet soils during flowering and fruit setting. Drought periods are needed prior to blossom showers.',
        'desc': 'Coffee is a shade-loving plantation crop grown in hilly tropical highlands. Requires a wet season followed by a dry period.',
        'max_yield': 1.5
    }
}

# Default global values of means/stds computed from standard dataset to fall back if file is missing
DEFAULT_STATS = {
    'N': {'mean': 50.55, 'std': 36.91},
    'P': {'mean': 53.36, 'std': 32.98},
    'K': {'mean': 48.15, 'std': 50.65},
    'temperature': {'mean': 25.61, 'std': 5.06},
    'humidity': {'mean': 71.48, 'std': 22.26},
    'ph': {'mean': 6.47, 'std': 0.77},
    'rainfall': {'mean': 103.46, 'std': 54.95}
}

def load_ml_components():
    """Loads pickled model, scaler, and label encoder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return None, None, None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

def predict_crop_recommendation(n, p, k, temperature, humidity, ph, rainfall):
    """
    Predicts the recommended crop and details based on inputs.
    Returns a dict with: recommended_crop, confidence_score, yield_potential, details.
    """
    model, scaler, label_encoder = load_ml_components()
    if model is None:
        return {
            'error': 'ML models are not trained yet. Please train the model via the Admin page first.'
        }

    # Format inputs
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)
    
    # Predict label
    pred_encoded = model.predict(scaled_features)[0]
    crop_label = label_encoder.inverse_transform([pred_encoded])[0]
    
    # Predict probability/confidence
    try:
        prob = model.predict_proba(scaled_features)[0]
        confidence = float(prob[pred_encoded])
    except (AttributeError, IndexError):
        # Fallback for models without predict_proba
        confidence = 1.0

    # Retrieve crop metadata
    crop_info = CROP_METADATA.get(crop_label, {
        'name': crop_label.capitalize(),
        'season': 'Kharif / Rabi',
        'fertilizer': 'NPK general application.',
        'irrigation': 'Moderate irrigation.',
        'desc': 'No description available.',
        'max_yield': 2.0
    })

    # Estimate suitability/compatibility score
    suitability = assess_crop_suitability(crop_label, n, p, k, temperature, humidity, ph, rainfall)
    compatibility_score = suitability['compatibility_score']
    
    # Yield potential scales with compatibility
    yield_est = round(crop_info['max_yield'] * (compatibility_score / 100.0), 2)
    
    return {
        'crop_label': crop_label,
        'crop_name': crop_info['name'],
        'confidence_score': round(confidence * 100, 2),
        'yield_potential': yield_est,
        'season': crop_info['season'],
        'fertilizer': crop_info['fertilizer'],
        'irrigation': crop_info['irrigation'],
        'desc': crop_info['desc'],
        'compatibility_score': compatibility_score,
        'suitability_rating': suitability['suitability_rating']
    }

def assess_crop_suitability(crop_name, n, p, k, temperature, humidity, ph, rainfall):
    """
    Evaluates how suitable a specific crop is for the given environmental conditions.
    Computes a compatibility percentage using multivariate Gaussian distance (Z-scores).
    """
    # Try to load means and standard deviations from the dataset to be dynamically accurate
    try:
        df = pd.read_csv(DATASET_PATH)
        crop_data = df[df['label'] == crop_name]
        if len(crop_data) < 5:
            # Fallback to general stats if not enough crop samples
            raise Exception("Insufficient data for crop")
        
        means = crop_data.mean(numeric_only=True).to_dict()
        stds = crop_data.std(numeric_only=True).to_dict()
    except Exception:
        # High quality hardcoded values matching the centroids of standard crop distributions
        # if the dataset is currently unavailable
        df = None
        crop_profiles = {
            'rice': {'N': (80, 15), 'P': (47.5, 10), 'K': (40, 5), 'temp': (23.5, 3.0), 'hum': (85, 5), 'ph': (6.0, 0.5), 'rain': (240, 30)},
            'maize': {'N': (80, 15), 'P': (47.5, 10), 'K': (20, 5), 'temp': (24, 4.0), 'hum': (62.5, 5), 'ph': (6.25, 0.5), 'rain': (85, 15)},
            'chickpea': {'N': (40, 10), 'P': (67.5, 8), 'K': (80, 5), 'temp': (19, 2.0), 'hum': (17.5, 2.5), 'ph': (7.0, 0.5), 'rain': (80, 10)},
            'kidneybeans': {'N': (25, 10), 'P': (67.5, 8), 'K': (20, 5), 'temp': (20, 3.0), 'hum': (21.5, 2.5), 'ph': (5.75, 0.3), 'rain': (105, 20)},
            'pigeonpeas': {'N': (25, 10), 'P': (67.5, 8), 'K': (20, 5), 'temp': (26.5, 4.0), 'hum': (50, 10), 'ph': (6.0, 0.8), 'rain': (145, 30)},
            'mothbeans': {'N': (25, 10), 'P': (47.5, 8), 'K': (20, 5), 'temp': (28.5, 3.0), 'hum': (52.5, 8), 'ph': (6.25, 1.0), 'rain': (52.5, 15)},
            'mungbean': {'N': (25, 10), 'P': (47.5, 8), 'K': (20, 5), 'temp': (28.5, 2.0), 'hum': (85, 5), 'ph': (6.7, 0.3), 'rain': (47.5, 10)},
            'blackgram': {'N': (40, 10), 'P': (67.5, 8), 'K': (20, 5), 'temp': (30, 3.0), 'hum': (65, 5), 'ph': (7.0, 0.3), 'rain': (67.5, 10)},
            'lentil': {'N': (25, 10), 'P': (47.5, 8), 'K': (20, 5), 'temp': (22.5, 3.0), 'hum': (65, 5), 'ph': (6.4, 0.3), 'rain': (45, 8)},
            'pomegranate': {'N': (20, 10), 'P': (17.5, 5), 'K': (40, 5), 'temp': (21.5, 2.5), 'hum': (90, 3), 'ph': (6.5, 0.5), 'rain': (105, 5)},
            'banana': {'N': (100, 10), 'P': (82.5, 8), 'K': (50, 5), 'temp': (27.5, 2.0), 'hum': (80, 4), 'ph': (6.0, 0.3), 'rain': (100, 10)},
            'mango': {'N': (20, 10), 'P': (27.5, 6), 'K': (30, 4), 'temp': (31.5, 3.0), 'hum': (50, 4), 'ph': (5.75, 0.6), 'rain': (95, 10)},
            'grapes': {'N': (20, 10), 'P': (132.5, 8), 'K': (200, 5), 'temp': (25, 5.0), 'hum': (82.5, 3), 'ph': (5.75, 0.2), 'rain': (70, 5)},
            'watermelon': {'N': (100, 10), 'P': (17.5, 5), 'K': (50, 4), 'temp': (25.5, 1.5), 'hum': (85, 3), 'ph': (6.5, 0.3), 'rain': (50, 8)},
            'muskmelon': {'N': (100, 10), 'P': (17.5, 5), 'K': (50, 4), 'temp': (28.5, 1.5), 'hum': (92.5, 2), 'ph': (6.4, 0.2), 'rain': (25, 5)},
            'apple': {'N': (20, 10), 'P': (132.5, 8), 'K': (200, 5), 'temp': (22.5, 1.5), 'hum': (92.5, 2), 'ph': (6.0, 0.3), 'rain': (112.5, 8)},
            'orange': {'N': (20, 10), 'P': (17.5, 5), 'K': (10, 3), 'temp': (25, 5.0), 'hum': (92.5, 2), 'ph': (7.0, 0.6), 'rain': (110, 8)},
            'papaya': {'N': (50, 10), 'P': (57.5, 8), 'K': (50, 4), 'temp': (33.5, 4.0), 'hum': (92.5, 2), 'ph': (6.75, 0.2), 'rain': (245, 5)},
            'coconut': {'N': (20, 10), 'P': (17.5, 5), 'K': (30, 4), 'temp': (27.5, 2.0), 'hum': (95, 3), 'ph': (6.0, 0.3), 'rain': (177.5, 25)},
            'cotton': {'N': (120, 10), 'P': (47.5, 8), 'K': (20, 4), 'temp': (24, 1.5), 'hum': (80, 3), 'ph': (6.9, 0.5), 'rain': (80, 15)},
            'jute': {'N': (80, 10), 'P': (47.5, 8), 'K': (40, 4), 'temp': (25, 1.5), 'hum': (80, 5), 'ph': (6.5, 0.3), 'rain': (175, 15)},
            'coffee': {'N': (100, 10), 'P': (27.5, 8), 'K': (30, 4), 'temp': (24, 2.0), 'hum': (55, 3), 'ph': (6.75, 0.4), 'rain': (165, 15)}
        }
        profile = crop_profiles.get(crop_name, crop_profiles['rice'])
        means = {
            'N': profile['N'][0], 'P': profile['P'][0], 'K': profile['K'][0],
            'temperature': profile['temp'][0], 'humidity': profile['hum'][0],
            'ph': profile['ph'][0], 'rainfall': profile['rain'][0]
        }
        stds = {
            'N': profile['N'][1], 'P': profile['P'][1], 'K': profile['K'][1],
            'temperature': profile['temp'][1], 'humidity': profile['hum'][1],
            'ph': profile['ph'][1], 'rainfall': profile['rain'][1]
        }

    inputs = {
        'N': float(n),
        'P': float(p),
        'K': float(k),
        'temperature': float(temperature),
        'humidity': float(humidity),
        'ph': float(ph),
        'rainfall': float(rainfall)
    }

    # Compute Z-score distances
    z_squares = []
    assessment_report = {}
    missing_nutrients = []
    suggestions = []

    for param, value in inputs.items():
        mean_val = means[param]
        std_val = stds[param] if stds[param] > 0 else 1.0
        
        # Calculate Z-score
        z = (value - mean_val) / std_val
        z_squares.append(z ** 2)

        # Diagnose individual param
        lower_bound = mean_val - 1.5 * std_val
        upper_bound = mean_val + 1.5 * std_val

        status = "Optimal"
        alert = None
        
        if value < lower_bound:
            status = "Low"
            if param in ['N', 'P', 'K']:
                missing_nutrients.append(param)
        elif value > upper_bound:
            status = "High"

        assessment_report[param] = {
            'value': value,
            'mean': round(mean_val, 2),
            'status': status
        }

    # Calculate overall similarity score using a exponential decay distance function
    # Mean of squared Z-scores
    mean_z2 = np.mean(z_squares)
    # Map distance to 0 - 100% scale
    score = np.exp(-0.25 * mean_z2) * 100
    compatibility_score = round(max(0.0, min(100.0, score)), 2)

    # Suitability Rating Classifications
    if compatibility_score >= 80:
        rating = "Excellent"
    elif compatibility_score >= 60:
        rating = "Good"
    elif compatibility_score >= 40:
        rating = "Moderate"
    else:
        rating = "Poor"

    # Add customized suggestions
    if 'N' in missing_nutrients:
        suggestions.append("Apply nitrogenous fertilizers like Urea, Ammonium Nitrate, or incorporate leguminous cover crops (peas, beans) in your crop rotation.")
    if 'P' in missing_nutrients:
        suggestions.append("Apply phosphate-rich fertilizers such as Single Superphosphate (SSP), Diammonium Phosphate (DAP), or organic Bone Meal.")
    if 'K' in missing_nutrients:
        suggestions.append("Incorporate Potash fertilizers (Muriate of Potash - MOP, Potassium Sulfate) or wood ash to replenish Potassium levels.")

    # Soil pH suggestion
    ph_val = inputs['ph']
    ph_mean = means['ph']
    if ph_val < ph_mean - 0.7:
        suggestions.append(f"Soil is too acidic ({ph_val:.1f}) for {crop_name}. Consider adding agricultural lime (calcium carbonate) or dolomite to raise the pH to optimal (~{ph_mean:.1f}).")
    elif ph_val > ph_mean + 0.7:
        suggestions.append(f"Soil is too alkaline ({ph_val:.1f}) for {crop_name}. Apply agricultural sulfur, gypsum, or organic compost to lower the pH to optimal (~{ph_mean:.1f}).")

    # Rainfall suggestion
    rain_val = inputs['rainfall']
    rain_mean = means['rainfall']
    if rain_val < rain_mean - 40:
        suggestions.append(f"Rainfall ({rain_val:.0f} mm) is below optimal (~{rain_mean:.0f} mm). Ensure supplementary irrigation via drip systems or sprinklers.")
    elif rain_val > rain_mean + 50:
        suggestions.append(f"Water intake ({rain_val:.0f} mm) exceeds optimal (~{rain_mean:.0f} mm). Plan for raised beds and check drainage channels to avoid waterlogging and root asphyxiation.")

    # General suggestion if things are perfect
    if len(suggestions) == 0:
        suggestions.append("Soil and weather parameters match optimal criteria perfectly. Standard cultural operations and regular weeding will ensure high yield potential.")

    return {
        'crop': crop_name,
        'compatibility_score': compatibility_score,
        'suitability_rating': rating,
        'report': assessment_report,
        'missing_nutrients': missing_nutrients,
        'suggestions': suggestions
    }
