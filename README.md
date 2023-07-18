# Music-Recommendation Based on User's Facial Emotion
1.	Clone the Repository:
Clone the project repository from GitHub using the following command in your terminal or command prompt:
git clone https://github.com/SameerRaja2331/Music-Recommendation.git

2.	Set up the Environment:
You can use the pip package manager to install the dependencies by running:
pip install -r requirements.txt

3.	Download the trained model from the drive link provided below and include it in the **project root folder**.
https://drive.google.com/file/d/1hkqPf2ZJC5wJVvu6kZrCLTO54zH8DmhC/view?usp=sharing

4.	Obtain Spotify API Credentials:
To integrate Spotify into the project, you'll need to obtain Spotify API credentials (Client ID and Client Secret) as described earlier. Create a Spotify Developer Account, register a Spotify application, and obtain the required credentials. 

5.	Update the Code with API Credentials:
In your project code, locate the section where Spotify API authentication is set up. Replace <your_client_id> and <your_client_secret> with your actual Spotify API credentials.

6.	Similarly if you want make use of youtube api, go to google cloud console. Enable the youtube data api v3 and obtain the API credentials. Then in your project code replace the <api_key> with your actual youtube API credentials.

7.	Run the Project:
Once the environment is set up and the code is updated with the necessary credentials, you can run the project. Execute the main script file that is main.py using the command -
python main.py or you can use any IDE and run the main file.

8.	Test the Project:
Test the project by using facial expressions in front of your camera or webcam. The system should detect your facial expressions and recommend appropriate music playlists based on your emotions.
