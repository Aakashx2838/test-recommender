import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data - replace this with your actual dataset
courses_data = pd.read_csv("./dataset/course.csv")

user_enrollments = pd.read_csv("./dataset/enrollments.csv")

# Create a user-item matrix for collaborative filtering (KNN)
user_course_matrix = user_enrollments.pivot_table(
    index="user_id", columns="course_id", values="rating", fill_value=0
)

# Train KNN model for user-based collaborative filtering
knn_model_user = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model_user.fit(user_course_matrix)

# Create a course-item matrix for content-based filtering
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
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
    return courses_data["title"]


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
user_id = 900
course_title = "DP-900 Azure Data Fundamentals Exam Prep In One Day"
recommendations = hybrid_recommendations(
    user_id, course_title, knn_model_user, user_course_matrix
)
print(recommendations)
