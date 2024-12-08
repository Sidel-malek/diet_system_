from flask import Flask, request, render_template, jsonify, session
import joblib
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors


app = Flask(__name__)
app.secret_key = 'malek key'  # Required for session management

more_details_button = False

# Load the pre-trained KNN model for supervised
knn = joblib.load('knn_supervised_model_.pkl')
# Load the vectorizer
vectorizer = joblib.load('supervise_vectorizer_.pkl')

# Load the pre-trained KNN model  for unsupervised
knn_similar =joblib.load('knn_unsupervised_model_.pkl')
  
preprocessor_similar =joblib.load('unsupervised_preprossesor_.pkl')

#scaler = joblib.load('scaler_diet.pkl')
kmeans = joblib.load('kmeans_diet.pkl')


# Load the dataset
data = pd.read_csv('cleaned_recipes_.csv')


def fetch_image(recipe_name , ingredients):
    url = "https://cse.google.com/cse.js?cx=b604b290643ae4f55"
    API_KEY = "AIzaSyCcPP5R23o7DdTVxCdLvBIwXKqm_ullXG4"
    SEARCH_ENGINE_ID = "b604b290643ae4f55"

    search_query = f"{recipe_name} {ingredients}"

    params = {
        "q": search_query,
        "cx": SEARCH_ENGINE_ID,
        "key": API_KEY,
        "searchType": "image",
        "num": 1
    }

    try:
        # Effectuer la requête
        response = requests.get(url, params=params)

        # Vérifier le code de statut HTTP
        if response.status_code != 200:
            print(f"Erreur HTTP: {response.status_code}")
            print(f"Message: {response.text}")
            return None

        # Vérifier si la réponse contient du JSON
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Erreur : La réponse ne contient pas de JSON valide.")
            print(f"Contenu brut de la réponse : {response.text}")
            return None

        # Vérifier si des résultats existent dans le JSON
        if "items" in data and len(data["items"]) > 0:
            image_url = data["items"][0]["link"]  # URL de la première image
            return image_url
        else:
            print("Aucune image trouvée pour cette recette.")
            return None

    except requests.RequestException as e:
        print(f"Erreur de requête : {e}")
        return None

@app.route('/')
def home():
    global recipes_limited
    recipes_limited = data.sample(n=6)
    return render_template("index.html", recipes=recipes_limited.to_dict("records"))




@app.route("/recommend/<int:recipe_id>")
def recommend(recipe_id):
    """
    Recommend similar recipes based on the selected recipe ID.
    """
    # Find the index of the selected recipe in the dataset
    selected_indexes = data.index[data["RecipeId"] == recipe_id].tolist()

    if not selected_indexes:
        return "Recipe not found", 404  # Handle the case when no recipe matches the given ID

    selected_index = selected_indexes[0]

    # Find the row for the selected recipe
    recipe_row = data.iloc[selected_index]
    recipe_row['Images'] = fetch_image(recipe_row['Name'],recipe_row['RecipeIngredientParts'])
    # Find similar recipes using the k-NN model
    distances, indices = knn_similar.kneighbors(preprocessor_similar.transform(data.iloc[[selected_index]]), n_neighbors=10)

    # Get the recommended recipes
    similar_recipes = data.iloc[indices[0]].to_dict("records")
    # Fetch images for each of the recommended recipes
    for recipe in similar_recipes:
        # Fetch the image URL for each recommended recipe
        recipe_name = recipe['Name']
        ingredients = recipe['RecipeIngredientParts']  # Join ingredients if it's a list
        recipe['Images'] = fetch_image(recipe_name, ingredients)  # Add image URL to the recipe dict
    return render_template("recommendations.html", selected_recipe=recipe_row, recommendations=similar_recipes)


################################################################################################

@app.route('/predict', methods=['POST'])
def predict():
    ingredients = request.form['ingredients']
    user_vector = vectorizer.transform([ingredients])
    prediction = knn.predict(user_vector)[0]

    # Get the most similar recipe indices
    _, indices = knn.kneighbors(user_vector)
    
    top_5_indices = [int(index) for index in indices[0][:6]]  # Convert to Python int
    session['remaining_recipes'] = top_5_indices  # Save indices in session
    session['prediction'] = prediction  # Save prediction in session for reuse

    return get_recipe(prediction)




@app.route('/next_recipe', methods=['POST'])
def next_recipe():
    prediction = session.get('prediction', 'Unknown')  # Retrieve stored prediction
    return get_recipe(prediction)

#############################################################################################

#Fonction pour calculer les besoins caloriques journaliers (formule de Mifflin-St Jeor)
def calculate_daily_calories(weight, height, age, sex, goal_weight):
    if sex == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    daily_calories = bmr + (goal_weight - weight) * 500 / abs(goal_weight - weight if goal_weight != weight else 1)
    return daily_calories

# Calculer les besoins en macronutriments
def calculate_macronutrients(daily_calories):
    protein_percentage = 0.15  # 15% des calories
    fat_percentage = 0.25      # 25% des calories
    carb_percentage = 0.60     # 60% des calories
    
    protein_grams = (daily_calories * protein_percentage) / 4
    fat_grams = (daily_calories * fat_percentage) / 9
    carb_grams = (daily_calories * carb_percentage) / 4
    
    return protein_grams, fat_grams, carb_grams

# Répartition des calories entre les repas
def distribute_calories(daily_calories):
    breakfast_calories = daily_calories * 0.25
    lunch_calories = daily_calories * 0.40
    dinner_calories = daily_calories * 0.35
    return breakfast_calories, lunch_calories, dinner_calories

import pandas as pd

def get_user_input():
    """Demande les informations nécessaires à l'utilisateur."""
    print("\nVeuillez fournir les informations suivantes :")
    try:
        weight = float(input("Votre poids actuel (kg) : "))
        height = float(input("Votre taille (cm) : "))
        age = int(input("Votre âge (ans) : "))
        sex = input("Votre sexe (Homme/Femme) : ").strip().lower()
        if sex not in ['homme', 'femme']:
            raise ValueError("Sexe invalide. Veuillez entrer 'Homme' ou 'Femme'.")
        goal_weight = float(input("Votre poids objectif (kg) : "))
    except ValueError as e:
        print(f"Erreur : {e}. Veuillez recommencer.")
        return get_user_input()
    
    return weight, height, age, sex, goal_weight

# Filtrer les recettes par type de repas
def get_recipes_for_meal(meal_type, recipes):
    return recipes[recipes['MealType'] == meal_type]


@app.route('/diet', methods=['GET', 'POST'])
def dietObjectif():
    # Obtenir les informations utilisateur
    if request.method == 'POST':
        # Récupérer les informations utilisateur
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        sex = request.form['sex']
        goal_weight = float(request.form['goal_weight'])

        # Calcul des besoins caloriques et macronutriments
        daily_calories = calculate_daily_calories(weight, height, age, sex, goal_weight)

        protein_grams, fat_grams, carb_grams = calculate_macronutrients(daily_calories)

        # Distribution des calories par repas
        breakfast_calories, lunch_calories, dinner_calories = distribute_calories(daily_calories)
    
        # Initialisation des listes de recettes
        breakfast_recipes = []
        lunch_recipes = []
        dinner_recipes = []

        # Recommandations par repas
        for meal, calories, recipes in zip(['Breakfast', 'Lunch', 'Dinner'], [breakfast_calories, lunch_calories, dinner_calories], [breakfast_recipes, lunch_recipes, dinner_recipes]):
            print(f"{meal.capitalize()}")
            print("-" * 80)
        
            if meal == 'Lunch' or meal == 'Dinner':
                meal_ = 'Dinner_Lunch'
            else:
                meal_ = 'Breakfast'
            
            query = np.array([[calories, fat_grams, protein_grams, carb_grams]])
            cluster = kmeans.predict(query)[0]
            data['Cluster'] = kmeans.predict(data[['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']])
            
            meal_recipes = get_recipes_for_meal(meal_, data)

            if meal_recipes.empty:
                print(f"Aucune recette trouvée pour {meal}.\n")
                continue

            # Recommander des recettes pour le cluster et afficher les résultats
            cluster_recipes = meal_recipes[meal_recipes['Cluster'] == cluster]

            # Appliquer KNN pour trouver les recettes les plus proches
            knn_diet = NearestNeighbors(n_neighbors=5)
            knn_diet.fit(cluster_recipes[['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']].values)
    
            distances, indices = knn_diet.kneighbors(query)

            # Extraire les recettes recommandées
            recommended_recipes = cluster_recipes.iloc[indices[0]]
            recipes.extend(recommended_recipes.to_dict(orient='records'))

        # Rendu de la page avec toutes les recettes
        return render_template(
            'diet.html',
            daily_calories=daily_calories,
            breakfast_calories=breakfast_calories,
            lunch_calories=lunch_calories,
            dinner_calories=dinner_calories,
            breakfast=breakfast_recipes,
            lunch=lunch_recipes,
            dinner=dinner_recipes
        )
    
    return render_template('diet.html')


##########################################################################################
def get_recipe(prediction):
    if 'remaining_recipes' not in session:
        return render_template(
            'index.html',
            prediction_text=f'No recipes available.'
        )

    # Get the first recipe index from the session
    recipe_index = session['remaining_recipes'].pop(0)  # Remove the first element
    session['remaining_recipes'].append(recipe_index)  # Add it to the end
    session.modified = True  # Notify Flask that the session has changed

    most_similar_recipe = data.iloc[recipe_index]
    recipe_name = most_similar_recipe['Name']
    
    # Make sure to join ingredient parts as a single string
    image_url = fetch_image(recipe_name, ' '.join(most_similar_recipe['RecipeIngredientParts']))

    # Recipe details to display
    recipe_details = {
        'Name': recipe_name,
        'CookTime': most_similar_recipe['CookTime'],
        'Images': image_url,
        'RecipeCategory': most_similar_recipe['RecipeCategory'],
        'RecipeIngredientQuantities': most_similar_recipe['RecipeIngredientQuantities'],
        'RecipeIngredientParts': most_similar_recipe['RecipeIngredientParts'],
        'AggregatedRating': f"{most_similar_recipe['AggregatedRating']:.2f}",
        'RecipeInstructions': most_similar_recipe['RecipeInstructions'],
        'RecipeServings': int(most_similar_recipe['RecipeServings']),
    }

    extended_details = {
        'Calories': f"{most_similar_recipe['Calories']:.2f}",
        'FatContent': f"{most_similar_recipe['FatContent']:.2f}",
        'SaturatedFatContent': f"{most_similar_recipe['SaturatedFatContent']:.2f}",
        'CholesterolContent': f"{most_similar_recipe['CholesterolContent']:.2f}",
        'SodiumContent': f"{most_similar_recipe['SodiumContent']:.2f}",
        'CarbohydrateContent': f"{most_similar_recipe['CarbohydrateContent']:.2f}",
        'FiberContent': f"{most_similar_recipe['FiberContent']:.2f}",
        'SugarContent': f"{most_similar_recipe['SugarContent']:.2f}",
        'ProteinContent': f"{most_similar_recipe['ProteinContent']:.2f}"
    }

    return render_template(
        'index.html',
        prediction_text=f'The recipe is: {prediction}',
        recipe_details=recipe_details,
        extended_details=extended_details,
        recipes=recipes_limited.to_dict("records")
    )

def find_similar_recipes(recipe_id, data, knn_model, preprocessor, n_neighbors=10):
    """
    Find similar recipes to a given recipe.
    """
    # Preprocess the recipe features
    recipe_features = preprocessor.transform(data.iloc[[recipe_id]])
    
    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(recipe_features, n_neighbors=n_neighbors)
    
    # Retrieve similar recipes
    similar_recipes = data.iloc[indices[0]].copy()
    similar_recipes['Distance'] = distances[0]
    
    return similar_recipes





if __name__ == "__main__":
    app.run(debug=True)
