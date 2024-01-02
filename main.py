import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data - replace this with your actual dataset
courses_data = pd.DataFrame(
    {
        "course_id": [1, 2, 3, 4, 5],
        "title": [
            "Python Basics",
            "Data Science Fundamentals",
            "Machine Learning in Python",
            "Web Development with Django",
            "Java Programming",
        ],
    }
)

user_enrollments = pd.DataFrame(
    {
        "user_id": [101, 102, 103, 104, 105],
        "course_id": [1, 2, 1, 3, 4],  # Sample enrollment data
        "rating": [5, 4, 5, 4, 3],  # Sample rating data
    }
)

# Create a user-item matrix for collaborative filtering (KNN)
user_course_matrix = user_enrollments.pivot_table(
    index="user_id", columns="course_id", values="rating", fill_value=0
)

# Train KNN model for user-based collaborative filtering
knn_model_user = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model_user.fit(user_course_matrix)

# Create a course-item matrix for content-based filtering
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
courses_data["description"] = [
    "Learn Python programming basics",
    "Explore fundamentals of data science",
    "Master machine learning with Python",
    "Build web applications with Django",
    "Java programming for beginners",
]
tfidf_matrix_courses = tfidf_vectorizer.fit_transform(courses_data["description"])

# Compute similarity scores using linear kernel
cosine_sim_courses = linear_kernel(tfidf_matrix_courses, tfidf_matrix_courses)


# Function to get user-based collaborative filtering recommendations
def knn_user_recommendations(user_id, knn_model, user_course_matrix):
    user_idx = user_enrollments.index[user_enrollments["user_id"] == user_id].tolist()[
        0
    ]
    distances, indices = knn_model.kneighbors(
        user_course_matrix.iloc[user_idx, :].values.reshape(1, -1), n_neighbors=5
    )
    return user_enrollments["course_id"].iloc[indices.flatten()[1:]]


# Function to get content-based recommendations
def content_based_recommendations(course_title, cosine_sim=cosine_sim_courses):
    idx = courses_data.index[courses_data["title"] == course_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Exclude the course itself
    course_indices = [i[0] for i in sim_scores]
    return courses_data["title"].iloc[course_indices]


# Function to get hybrid recommendations
def hybrid_recommendations(
    user_id,
    course_title,
    knn_model_user,
    user_course_matrix,
    cosine_sim=cosine_sim_courses,
):
    # Get user-based collaborative filtering recommendations
    user_cf_recommendations = knn_user_recommendations(
        user_id, knn_model_user, user_course_matrix
    )

    # Get content-based recommendations
    cb_recommendations = content_based_recommendations(course_title, cosine_sim)

    # Combine both recommendations
    hybrid_recommendations = set(user_cf_recommendations).union(set(cb_recommendations))

    return list(hybrid_recommendations)[:5]  # Limit to 5 recommendations


# Example usage
user_id = 101
course_title = "Python Basics"
recommendations = hybrid_recommendations(
    user_id, course_title, knn_model_user, user_course_matrix
)
print(recommendations)
