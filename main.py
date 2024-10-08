# streamlit_app.py

import streamlit as st
import openai
import json
import re
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
import numpy as np

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(
    api_key=api_key,
)


# Define the PlayerPositionPlotter class
class PlayerPositionPlotter:
    def __init__(self, pitch_length=120, pitch_width=80):
        """
        Initialize the PlayerPositionPlotter with pitch dimensions.
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.pitch = Pitch(pitch_length=self.pitch_length, pitch_width=self.pitch_width,
                           line_color='white', pitch_color='#aabb97')

    def plot_player_positions(self, player_data):
        """
        Plots the players' positions based on input data.

        Args:
        player_data (dict): JSON-like dictionary containing player positions, ball, and other details.
        """
        # Initialize the pitch
        fig, ax = self.pitch.draw(figsize=(10, 7))

        # Extract player and ball data
        team_players = player_data.get('team_players', [])
        opponent_players = player_data.get('opponent_players', [])
        ball_position = player_data.get('ball', None)
        main_player = player_data.get('main_player', None)

        # Plot team players, opponents, and ball
        self._plot_players(team_players, ax, color='#4CAF50', label='Team Player', main_player=main_player)
        self._plot_players(opponent_players, ax, color='#FF5733', label='Opponent Player')

        if ball_position:
            self._plot_ball(ax, ball_position)

        # Optional: Set the title (can be customized based on input)
        plt.title("Player and Ball Positions", fontsize=16, color='#34495e', weight='bold')
        plt.tight_layout()
        return fig

    def _plot_players(self, players, ax, color, label, main_player=None):
        """
        Private method to plot multiple players on the pitch.

        Args:
        players (list): List of player dictionaries containing position information.
        ax: The axis to plot the players on.
        color (str): Color of the players for the respective team or opponent.
        label (str): Label for the player type (team or opponent).
        main_player (list): The position of the main player to highlight.
        """
        for player in players:
            position = player.get('position')
            is_main = (position == main_player) if main_player else False
            self._plot_player(ax, position, color, is_main=is_main, label=label)

    def _plot_player(self, ax, position, color, is_main=False, label=None):
        """
        Private method to plot a single player at a specific position.

        Args:
        ax: The axis to plot the player on.
        position (list): The [x, y] position on the pitch.
        color (str): Color of the player marker.
        is_main (bool): Whether the player is the main player.
        label (str): Player type (team or opponent).
        """
        marker_size = 300
        edge_color = 'gold' if is_main else 'black'
        ax.scatter(position[0], position[1], s=marker_size, color=color, edgecolor=edge_color, lw=2.5, label=label,
                   zorder=3)

    def _plot_ball(self, ax, position):
        """
        Private method to plot the ball on the pitch.

        Args:
        ax: The axis to plot the ball on.
        position (list): The [x, y] position of the ball on the pitch.
        """
        # Plot the ball as a circle
        ball = Circle((position[0], position[1]), radius=1, color='white', zorder=4)
        ax.add_patch(ball)


# Define prompts
prompt_question = """You are an AI model tasked with creating a single soccer-related quiz question in the French 
language. The question should focus on soccer tactics, rules, or scenarios, and the answers will evaluate the 
player's thinking across four key axes: conscience tactique (tactical awareness), compétences techniques (technical 
skills), mentalité (mindset), and attributs physiques (physical attributes). Each option should be associated with a 
score (out of 10) for each of the four axes. The scores have to be precise. There is no correct or incorrect answer, 
but the scores should reflect different emphases based on the player's decisions. Ensure that your output strictly 
adheres to the following optimized JSON format, wrapped within a <JSON></JSON> tag, without additional formatting: {{ 
"question": "<A detailed soccer-related question in French>", "options": [ {{ "text": "<Option 1>", "scores": [ {{
"axis": "conscience_tactique", "score": <score_out_of_10>}}, {{"axis": "competences_techniques", 
"score": <score_out_of_10>}}, {{"axis": "mentalite", "score": <score_out_of_10>}}, {{"axis": "attributs_physiques", 
"score": <score_out_of_10>}} ] }}, ... ] }} Ensure the JSON is formatted correctly and output only the JSON without 
any additional text."""

prompt_position_template = """You are an AI model tasked with generating player positions and coordinates for a 
soccer scenario based on a quiz generated by another agent. Follow these steps carefully to ensure precision:

Step 1: Understand the context. The quiz question is related to soccer tactics, rules, or scenarios. The aim is to 
illustrate the scenario clearly on a soccer pitch using player positions. Consider the information provided in the 
question and options and align the positions with the given scenario.

Step 2: Define the pitch dimensions. The soccer pitch dimensions are 120 (coordinates from 0 (left) to 120 (right)) 
for the x-axis and 80 (from 0 (top) to 80 (bottom)) for the y-axis. This will help in placing the players 
appropriately on the field. Make sure the coordinates respect these boundaries and align logically with the scenario. 
Calculate the key stadium areas, like the penalty area, corners, attacking and defending positions, half spaces, 
goalkeeper position... these will help you be more conscious about the stadium dimensions.

Step 3: Position the main player. Place the main player at a position that reflects their key role in the scenario. 
Think about whether this player is attacking or defending, and place them accordingly. Make sure to assign the main 
player a unique position in the format: {{"position": [x_main, y_main]}}.

Step 4: Place the team players (5 players, including the main player). Distribute the remaining 4 players from the 
main player’s team around the pitch based on the scenario. These players should be positioned strategically to 
reflect typical game dynamics, such as positioning during an attack, defense, or counterattack.

Step 5: Position the opponent players (5 players, including the goalkeeper). Place the defending team's players in 
appropriate positions to counter the team with the ball. Ensure that one of the players is clearly positioned as the 
goalkeeper, staying close to the goal. The other 4 opponent players should be positioned according to the game flow.

Step 6: Place the ball. The ball should be positioned near the main player, reflecting its role in the scenario. 
Ensure that the ball’s coordinates are logical in relation to the main player's position.

Step 7: Output in JSON format. Ensure IMPERATIVELY the JSON is formatted correctly, wrap the json output between the two tags: 
<JSON></JSON> without using any extra formatting, avoid using `\\` in the json.

Once the player positions and ball location are determined, output the following in JSON format:
{{
  "coordinates": {{
    "team_players": [
      {{"position": [x1, y1]}},
      ...
    ],
    "opponent_players": [
      {{"position": [x6, y6]}},
      ...
    ],
    "main_player": [x_main, y_main],
    "ball": [ball_x, ball_y]
  }}
}}
context: {question_context} {options_context}
"""

prompt_correct = """You are an expert in soccer data analysis and visualization tasked with evaluating and correcting 
the positions of players generated by a previous agent. Your role is to ensure that the positions are tactically 
sound, precise, and relevant to the soccer scenario. Follow these steps carefully to provide a critical assessment 
and any necessary corrections:

Step 1: Analyze the game scenario. Start by understanding the soccer scenario described in the quiz question. Assess 
whether the positions of the main player, team players, and opponent players logically reflect the scenario. Is the 
scenario an attack, defense, or counterattack? Based on this, critically evaluate whether the player positions make 
sense in terms of typical soccer strategies and formations.

Step 2: Define the pitch dimensions. The soccer pitch dimensions are 120 (coordinates from 0 (left) to 120 (right)) 
for the x-axis and 80 (from 0 (top) to 80 (bottom)) for the y-axis. This will help in placing the players 
appropriately on the field. Make sure the coordinates respect these boundaries and align logically with the scenario. 
Calculate the key stadium areas, like the penalty area, corners, attacking and defending positions, half spaces, 
goalkeeper position... these will help you be more conscious about the stadium dimensions.

Step 3: Evaluate the main player’s position. Focus on the main player's position first. Does it accurately represent 
their role in the scenario? For example, if the scenario involves a counterattack, the main player should be 
positioned further forward with more space behind them. If the main player is defending, they should be closer to 
their own goal. Critique the precision and relevance of this position and make adjustments if needed. It's normal 
that you will find the main_player overlapping with a team_player; it's because they are the same, it's just the way 
of how we referencing the main_player by position.

Step 4: Assess the team players' positions. Next, examine the positioning of the other 4 team players. Consider 
whether these players are supporting the main player effectively given the context. Are they spread out correctly for 
an attack, maintaining proper spacing, or compact in defense? Make sure their positions reflect typical soccer 
strategies and correct any tactical misalignments.

Step 5: Review the opponent players’ positions. Evaluate the positioning of the 5 opponent players, especially the 
goalkeeper. The goalkeeper should be close to the goal and well-positioned to defend. Check if the opponent players 
are placed to effectively counter the main player’s team. If they are defending, are they positioned to block passing 
lanes or apply pressure? Adjust their positions as necessary based on soccer tactics. Check if the players are 
overlapping with each other; we don't need two dots on the top of each other.

Step 6: Critically analyze spacing and movement potential. Assess the spacing between the players. Is there too much 
or too little space between them for the scenario described? For example, in a counterattack, attackers should be 
spaced to exploit gaps, while defenders should cover key areas. Think about player movement potential: are players 
positioned in a way that allows them to transition effectively if the ball is passed or intercepted? Make tactical 
adjustments to improve flow and realism.

Step 7: Ensure ball positioning makes tactical sense. Evaluate whether the ball’s position is logical in relation to 
the main player and the overall scenario. For instance, in an attacking scenario, the ball should be near the main 
player, but far from the defenders. In a defensive setup, it should reflect the opposition’s pressure. Adjust the 
ball’s coordinates to reflect accurate game dynamics.

Step 8: Correct and finalize the positions in JSON format. Verify that the JSON is formatted correctly, and wrap the json 
output between the two tags: <JSON></JSON> without using any extra formatting (like ```json per example), avoid using 
`\\` in the json. After making all necessary adjustments to the player and ball positions, output the corrected data 
in JSON format as follows: {{ "coordinates": {{ "team_players": [ {{"position": [x1, y1]}}, {{"position": [x2, y2]}}, 
{{"position": [x3, y3]}}, {{"position": [x4, y4]}}, {{"position": [x5, y5]}} ], "opponent_players": [ {{"position": [
x6, y6]}}, {{"position": [x7, y7]}}, {{"position": [x8, y8]}}, {{"position": [x9, y9]}}, {{"position": [x10, 
y10]}} ], "main_player": [x_main, y_main], "ball": [ball_x, ball_y] }} }} Ensure all positions are precise and 
relevant, adhering strictly to the tactical scenario. question and answers: {question_context} {options_context} 
positions: {generated_positions}"""

# Function to generate a single soccer quiz question
def generate_single_soccer_quiz_question(prompt, api_key=api_key, model="gpt-4o"):
    openai.api_key = api_key

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    model_resp = response.choices[0].message.content.strip()
    print(model_resp)

    # Use regular expression to extract content between <JSON> tags
    json_match = re.search(r'<JSON>(.*?)</JSON>', model_resp, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()

        # Remove any comments like `//` using regex
        json_str = re.sub(r'//.*', '', json_str)  # This removes any inline comments starting with //

        try:
            quiz_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
    else:
        print("No JSON data found between <JSON> tags.")
        return None

    return quiz_data





# Function to plot the radar chart
def plot_radar_chart(total_scores):
    # Data for the radar chart
    labels = ['Conscience Tactique', 'Competences Techniques', 'Mentalite', 'Attributs Physiques']
    stats = [
        total_scores['conscience_tactique'],
        total_scores['competences_techniques'],
        total_scores['mentalite'],
        total_scores['attributs_physiques']
    ]

    # Number of variables we're plotting
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop" by appending the start value to the end
    stats += stats[:1]
    angles += angles[:1]

    # Set up the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='#FF5722', alpha=0.25)
    ax.plot(angles, stats, color='#FF5722', linewidth=2)

    # Add labels to the axes
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='#4CAF50', size=12)

    # Set the max limit for the chart (you can adjust based on your scoring system)
    ax.set_ylim(0, 10)

    # Show the radar chart
    return fig


def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Quiz Tactique de Football</h1>",
                unsafe_allow_html=True)

    # Initialize session state variables
    if 'questions' not in st.session_state:
        st.session_state.questions = []
        st.session_state.current_question = 0
        st.session_state.user_answers = []
        st.session_state.quiz_generated = False
        st.session_state.quiz_completed = False

    # User selects number of questions
    st.write("### Combien de questions souhaitez-vous générer ?")
    num_questions = st.number_input("", min_value=1, max_value=20, value=5)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Generate Quiz Questions
    if st.button("Générer des questions", key="generate_quiz") or st.session_state.quiz_generated:
        if not st.session_state.quiz_generated:
            st.session_state.questions = []
            st.session_state.quiz_completed = False
            status_placeholder = st.empty()

            for i in range(num_questions):
                with st.spinner(f"Génération de la question {i + 1}..."):
                    quiz_data = generate_single_soccer_quiz_question(prompt_question)
                    if quiz_data:
                        status_placeholder.text(f"Question {i + 1} générée avec succès.")
                        question_text = quiz_data['question']
                        options = quiz_data['options']

                        # Prepare context for position prompt (Agent 2)
                        question_context = f"\"question\": \"{question_text}\","
                        options_context = "\"options\": [\n"
                        for option in options:
                            options_context += f"  {{\n    \"text\": \"{option['text']}\"\n  }},\n"
                        options_context += "]"

                        prompt_position = prompt_position_template.format(
                            question_context=question_context,
                            options_context=options_context
                        )

                        with st.spinner(f"Génération des positions de joueur pour la question {i + 1}..."):
                            positions_data = generate_single_soccer_quiz_question(prompt_position)

                        if positions_data:
                            status_placeholder.text(f"Positions de joueur pour la question {i + 1} générées.")
                            # Now correct the positions using Agent 3
                            generated_positions = json.dumps(positions_data['coordinates'])
                            prompt_correct_positions = prompt_correct.format(
                                question_context=question_context,
                                options_context=options_context,
                                generated_positions=generated_positions
                            )

                            with st.spinner(f"Correction des positions de joueur pour la question {i + 1}..."):
                                corrected_positions_data = generate_single_soccer_quiz_question(
                                    prompt_correct_positions)

                            if corrected_positions_data:
                                status_placeholder.text(f"Positions corrigées pour la question {i + 1}.")
                                quiz_data['coordinates'] = corrected_positions_data['coordinates']
                                st.session_state.questions.append(quiz_data)
                            else:
                                status_placeholder.error(
                                    f"Échec de la correction des positions pour la question {i + 1}.")
                        else:
                            status_placeholder.error(f"Échec de la génération des positions pour la question {i + 1}.")
                    else:
                        status_placeholder.error(f"Échec de la génération de la question {i + 1}.")

                status_placeholder.empty()

            st.session_state.quiz_generated = True
        st.markdown("<hr>", unsafe_allow_html=True)

        # Display the current question
        if not st.session_state.quiz_completed:
            if st.session_state.current_question < len(st.session_state.questions):
                q_data = st.session_state.questions[st.session_state.current_question]
                st.markdown(f"<h3 style='color: #FF5722;'>Question {st.session_state.current_question + 1}</h3>",
                            unsafe_allow_html=True)
                st.write(q_data['question'])

                # Plot the pitch and positions
                plotter = PlayerPositionPlotter(pitch_length=120, pitch_width=80)
                scenario_data = {
                    "team_players": q_data["coordinates"]["team_players"],
                    "opponent_players": q_data["coordinates"]["opponent_players"],
                    "main_player": q_data["coordinates"]["main_player"],
                    "ball": q_data["coordinates"]["ball"]
                }

                with st.spinner("Affichage des positions de joueurs..."):
                    fig = plotter.plot_player_positions(scenario_data)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    st.image(buf)

                # Display options
                options = q_data['options']
                option_texts = [opt['text'] for opt in options]
                user_choice = st.radio("Choisissez votre action :", option_texts,
                                       key=f"question_{st.session_state.current_question}")

                # Add Previous and Next/See Results buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.session_state.current_question > 0:
                        if st.button("Question Précédente", key="previous_question"):
                            st.session_state.current_question -= 1
                            if len(st.session_state.user_answers) > st.session_state.current_question:
                                st.session_state.user_answers.pop()
                            st.rerun()

                with col2:
                    if st.session_state.current_question == len(st.session_state.questions) - 1:
                        if st.button("Voir les résultats", key="see_results"):
                            st.session_state.user_answers.append(user_choice)
                            st.session_state.quiz_completed = True
                            st.rerun()
                    else:
                        if st.button("Question Suivante", key="next_question"):
                            st.session_state.user_answers.append(user_choice)
                            st.session_state.current_question += 1
                            st.rerun()

        else:
            # Quiz has been completed, calculate scores
            total_scores = {
                "conscience_tactique": 0,
                "competences_techniques": 0,
                "mentalite": 0,
                "attributs_physiques": 0
            }

            with st.spinner("Calcul des résultats..."):
                for idx, answer in enumerate(st.session_state.user_answers):
                    q_data = st.session_state.questions[idx]
                    for option in q_data['options']:
                        if option['text'] == answer:
                            for score in option['scores']:
                                total_scores[score['axis']] += score['score']
                            break

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #4CAF50;'>Résultats du Quiz</h3>", unsafe_allow_html=True)

            # Plot radar chart for the results
            fig = plot_radar_chart(total_scores)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

            # Reset the session state for a new quiz
            if st.button("Recommencer le Quiz", key="restart_quiz"):
                st.session_state.questions = []
                st.session_state.current_question = 0
                st.session_state.user_answers = []
                st.session_state.quiz_generated = False
                st.session_state.quiz_completed = False
                st.rerun()


if __name__ == "__main__":
    main()