from constants import STYLE_PRESETS, IMAGE_SUGGESTIONS, DESCRIPTION_SUGGESTIONS

import pendulum
import requests
from PIL import Image
import io
import os
import time
import openai
from openai import OpenAI

# from anthropic import (
#     Anthropic,
#     HUMAN_PROMPT,
#     AI_PROMPT,
#     APIError,
#     RateLimitError as AnthropicRateLimitError,
# )
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()


# Function to get environment variable and raise ValueError if not set
def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} is not set in the .env file")
    return value


# Get API keys from .env file
# OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = get_env_variable("ANTHROPIC_API_KEY")
STABILITY_API_KEY = get_env_variable("STABILITY_API_KEY")

# Initialize clients with API keys
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


class RateLimitError(Exception):
    pass


class APIError(Exception):
    def __init__(self, message, request=None):
        self.message = message
        self.request = request
        super().__init__(self.message)


class Kou:
    def __init__(
        self,
        kanji,
        romanji,
        english,
        start_date,
        end_date,
        description=None,
        images=None,
    ):
        self.kanji = kanji
        self.romanji = romanji
        self.english = english
        self.start_date = start_date
        self.end_date = end_date
        self.description = description
        self.images = images or []

    def generate_description(self, llm_type="gpt3", custom_prompt=""):
        """Generate description using specified LLM"""
        return
        try:
            if llm_type == "gpt3":
                self.description = self._generate_gpt3_description(custom_prompt)
            elif llm_type == "claude":
                self.description = self._generate_claude_description(custom_prompt)
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            return self.description
        except (APIError, RateLimitError) as e:
            print(f"Error generating description: {str(e)}")
            return None

    def _generate_description_prompt(self, custom_prompt=""):
        base_prompt = f"Describe the Japanese micro-season '{self.english}' ({self.kanji}) which occurs from {self.start_date} to {self.end_date}."
        if custom_prompt:
            return f"{base_prompt} {custom_prompt}"
        return base_prompt

    def _generate_gpt3_description(self, custom_prompt=""):
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                prompt = self._generate_description_prompt(custom_prompt)
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that describes Japanese micro-seasons.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RateLimitError("Rate limit exceeded for GPT-3 API")
            except openai.APIError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"GPT-3 API error: {str(e)}")
                    return None

    def _generate_claude_description(self, custom_prompt=""):
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                prompt = self._generate_description_prompt(custom_prompt)
                system_message = (
                    "You are an AI assistant specializing in Japanese seasons and culture. "
                    "Provide detailed and accurate information about the 72 seasons of Japan."
                )

                if hasattr(anthropic_client, "messages"):
                    # New Claude-3 API
                    response = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=1000,
                        temperature=0,  # Set temperature to 0 for deterministic output
                        system=system_message,
                        messages=[
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                    )
                    return response.content[0].text.strip()
                else:
                    # Fallback to old Claude-2 API
                    full_prompt = f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
                    response = anthropic_client.completions.create(
                        model="claude-2",
                        prompt=full_prompt,
                        max_tokens_to_sample=300,
                        stop_sequences=[HUMAN_PROMPT],
                    )
                    return response.completion.strip()

            except AnthropicRateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RateLimitError("Rate limit exceeded for Claude API")
            except APIError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Claude API error: {str(e)}")
                    return None

    @staticmethod
    def get_description_prompt_suggestions():
        """Return a list of prompt suggestions for description generation"""
        return DESCRIPTION_SUGGESTIONS

    # TODO: image_to_video https://platform.stability.ai/docs/api-reference#tag/Image-to-Video
    def generate_image(
        self,
        ai_model="stable_diffusion",
        style="realistic",
        custom_prompt="",
    ):
        """Generate image using specified AI model and style"""
        try:
            # Initialize model parameters
            model_params = {
                "model": ai_model,
                "style": style,
                "base_prompt": self.english,
                "custom_prompt": custom_prompt,
                "temperature": 0,  # Example parameter, adjust as needed
                "max_tokens": 1000,  # Example parameter, adjust as needed
                "seed": 42,  # Consistent seed for reproducibility
            }

            prompt = self._generate_image_prompt(style, custom_prompt)
            print(f"prompt: {prompt}")
            print(f"model_params: {model_params}")
            if ai_model == "stable_diffusion":
                # Generate image with the updated prompt and save to a dynamic filename
                image = self._generate_stable_diffusion_image(prompt, model_params)
            elif ai_model == "dall_e":
                image = self._generate_dall_e_image(prompt, model_params)
            else:
                raise ValueError(f"Unsupported AI model: {ai_model}")

            if image:
                self.images.append(image)
                filename = f"{self.english.replace(' ', '_')}"  # Replace spaces with underscores for the filename
                print(f"filename: {filename}")
                if filename:
                    self._save_image(image, ai_model, prompt, model_params, filename)
                return image
            return None
        except (APIError, RateLimitError) as e:
            print(f"Error generating image: {str(e)}")
            return None

    def _generate_stable_diffusion_image(self, prompt, model_params):
        """Generate an image using the Stable Diffusion Core API."""
        max_retries = 3
        retry_delay = 5  # seconds

        # Extract parameters from model_params
        seed = model_params.get("seed", None)  # Default to None if not provided
        output_format = model_params.get(
            "output_format", "png"
        )  # Default to png if not provided

        # Prepare the payload for the Stable Image Core endpoint
        payload = {
            "prompt": prompt,
            "seed": seed,
            "output_format": output_format,
            # Add any other optional parameters as needed
            # TODO: aspect_ratio; negative_prompt; style_preset
        }

        # Set the headers, including the API key
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",  # Replace with your actual API key
            "Accept": "image/*",  # Change to "application/json" if you want base64 response
        }

        for attempt in range(max_retries):
            try:
                # Make the POST request to the Stable Image Core API with multipart/form-data
                response = requests.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/core",
                    headers=headers,
                    files={
                        "prompt": (None, payload["prompt"]),  # Send prompt as form data
                        "seed": (None, str(payload["seed"])),  # Send seed if provided
                        "output_format": (
                            None,
                            payload["output_format"],
                        ),  # Send output format
                        # Add any other parameters as needed
                    },
                )

                # Check if the response is successful
                if response.status_code == 200:
                    # If you set Accept to "image/*", the response will be the image
                    return Image.open(io.BytesIO(response.content))
                else:
                    raise APIError(f"Stable Diffusion API error: {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise APIError(
                        f"Failed to generate image after {max_retries} attempts."
                    )

    def _generate_dall_e_image(self, prompt, model_params):
        """Generate an image using the DALL-E model."""
        max_retries = 3
        retry_delay = 5  # seconds

        # Extract parameters from model_params
        n_images = model_params.get("n_images", 1)  # Default to 1 if not provided
        size = model_params.get("size", "1024x1024")  # Default size if not provided
        seed = model_params.get("seed", None)  # Default to None if not provided

        for attempt in range(max_retries):
            try:
                # Call the DALL-E API with the specified parameters
                response = client.images.generate(
                    prompt=prompt,
                    n=n_images,
                    size=size,
                    seed=seed,  # Include the seed for consistent results
                )
                image_url = response.data[0].url
                return Image.open(requests.get(image_url, stream=True).raw)
            except client.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RateLimitError("Rate limit exceeded for DALL-E API")
            except client.APIError as e:
                raise APIError(f"DALL-E API error: {str(e)}")

    def _generate_image_prompt(self, style, custom_prompt=""):
        """Generate a prompt for image creation based on the Kou, style, and custom prompt"""
        base_prompt = f"Primary visual focus on Japanese micro-season '{self.english}'"
        # Get the style prompt based on the provided style
        style_prompt = STYLE_PRESETS.get(style, "realistic")
        full_prompt = f"{style_prompt}: {base_prompt}" + (
            f", with secondary elements: {custom_prompt}" if custom_prompt else ""
        )
        return full_prompt

    def _save_image(self, image, ai_model, prompt, params, filename):
        """Save the generated image with timestamp, model information, and parameters"""
        # Generate a UUID for the image name
        image_uuid = str(uuid.uuid4())

        # Determine the file extension
        file_extension = (
            os.path.splitext(filename)[1]
            if filename.endswith((".png", ".jpg", ".jpeg"))
            else ".png"
        )

        # Create the full filename with UUID
        timestamp_dateonly = pendulum.now("Asia/Tokyo").format("YYYYMMDD")
        full_filename = f"images/{ai_model}-{filename.lower()}-{timestamp_dateonly}_{image_uuid}{file_extension}"

        # Save the image
        image.save(full_filename)
        print(f"Image saved as: {full_filename}")

        # Save the prompt, model information, and parameters to prompts.txt
        with open("history.txt", "a") as f:
            f.write(f"UUID: {image_uuid}\n")
            timestamp_datetime = pendulum.now("Asia/Tokyo").format(
                "YYYY-MM-DD HH:mm:ss zz"
            )
            f.write(f"Timestamp: {timestamp_datetime}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Model: {ai_model}\n")
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"Saved Image: {full_filename}\n")
            f.write("\n")  # Add a newline for better separation between entries

    @staticmethod
    def get_image_prompt_suggestions():
        """Return a list of prompt suggestions for image generation"""
        return IMAGE_SUGGESTIONS

    # OTHER DESCRIPTIVE METHODS
    def get_start_date(self):
        return self.start_date

    def get_end_date(self):
        return self.end_date

    def get_next_kou(self):
        for sekki in SEKKI_DICT:
            for i, kou in enumerate(sekki.kous):
                if kou == self:
                    if i < len(sekki.kous) - 1:
                        return sekki.kous[i + 1]
                    else:
                        next_term_index = (SEKKI_DICT.index(sekki) + 1) % len(
                            SEKKI_DICT
                        )
                        return SEKKI_DICT[next_term_index].kous[0]
        return None

    def get_previous_kou(self):
        for sekki in SEKKI_DICT:
            for i, kou in enumerate(sekki.kous):
                if kou == self:
                    if i > 0:
                        return sekki.kous[i - 1]
                    else:
                        prev_term_index = (SEKKI_DICT.index(sekki) - 1) % len(
                            SEKKI_DICT
                        )
                        return SEKKI_DICT[prev_term_index].kous[-1]
        return None

    def __str__(self):
        return f"{self.english} ({self.kanji}) - {self.romanji} [{self.start_date}...{self.end_date}]"


class Term:
    def __init__(self, name, desc, kous):
        self.name = name
        self.desc = desc
        self.kous = kous
        self.start_date = self.get_start_date()
        self.end_date = self.get_end_date()

    def get_start_date(self):
        # Assuming kous is a list of dictionaries, each with a 'start_date' key
        first_kou = self.kous[0]
        return first_kou.get_start_date()

    def get_end_date(self):
        # Assuming kous is a list of dictionaries, each with an 'end_date' key
        last_kou = self.kous[-1]
        return last_kou.get_end_date()

    def get_next_term(self):
        current_index = SEKKI_DICT.index(self)
        return SEKKI_DICT[(current_index + 1) % len(SEKKI_DICT)]

    def get_previous_term(self):
        current_index = SEKKI_DICT.index(self)
        return SEKKI_DICT[(current_index - 1) % len(SEKKI_DICT)]

    def __str__(self):
        return f"{self.name} ({self.desc}) [{self.get_start_date()}...{self.get_end_date()}]"


# Create a list of Kou objects for each season
KOU_DICT = {
    "Risshun": [
        Kou(
            "東風解凍",
            "Harukaze kōri o toku",
            "East wind melts the ice",
            pendulum.date(2024, 2, 4),
            pendulum.date(2024, 2, 8),
        ),
        Kou(
            "黄鶯睍睆",
            "Kōō kenkan su",
            "Bush warblers start singing in the mountains",
            pendulum.date(2024, 2, 9),
            pendulum.date(2024, 2, 13),
        ),
        Kou(
            "魚上氷",
            "Uo kōri o izuru",
            "Fish emerge from the ice",
            pendulum.date(2024, 2, 14),
            pendulum.date(2024, 2, 18),
        ),
    ],
    "Usui": [
        Kou(
            "土脉潤起",
            "Tsuchi no shō uruoi okoru",
            "Rain moistens the soil",
            pendulum.date(2024, 2, 19),
            pendulum.date(2024, 2, 23),
        ),
        Kou(
            "霞始靆",
            "Kasumi hajimete tanabiku",
            "Mist starts to linger",
            pendulum.date(2024, 2, 24),
            pendulum.date(2024, 2, 28),
        ),
        Kou(
            "草木萌動",
            "Sōmoku mebae izuru",
            "Grass sprouts, trees bud",
            pendulum.date(2024, 3, 1),
            pendulum.date(2024, 3, 5),
        ),
    ],
    "Keichitsu": [
        Kou(
            "蟄虫啓戸",
            "Sugomori mushito o hiraku",
            "Hibernating insects surface",
            pendulum.date(2024, 3, 6),
            pendulum.date(2024, 3, 10),
        ),
        Kou(
            "桃始笑",
            "Momo hajimete saku",
            "First peach blossoms",
            pendulum.date(2024, 3, 11),
            pendulum.date(2024, 3, 15),
        ),
        Kou(
            "菜虫化蝶",
            "Namushi chō to naru",
            "Caterpillars become butterflies",
            pendulum.date(2024, 3, 16),
            pendulum.date(2024, 3, 20),
        ),
    ],
    "Shunbun": [
        Kou(
            "雀始巣",
            "Suzume hajimete sukū",
            "Sparrows start to nest",
            pendulum.date(2024, 3, 21),
            pendulum.date(2024, 3, 25),
        ),
        Kou(
            "櫻始開",
            "Sakura hajimete saku",
            "First cherry blossoms",
            pendulum.date(2024, 3, 26),
            pendulum.date(2024, 3, 30),
        ),
        Kou(
            "雷乃発声",
            "Kaminari sunawachi koe o hassu",
            "Distant thunder",
            pendulum.date(2024, 3, 31),
            pendulum.date(2024, 4, 4),
        ),
    ],
    "Seimei": [
        Kou(
            "玄鳥至",
            "Tsubame kitaru",
            "Swallows return",
            pendulum.date(2024, 4, 5),
            pendulum.date(2024, 4, 9),
        ),
        Kou(
            "鴻雁北",
            "Kōgan kaeru",
            "Wild geese fly north",
            pendulum.date(2024, 4, 10),
            pendulum.date(2024, 4, 14),
        ),
        Kou(
            "虹始見",
            "Niji hajimete arawaru",
            "First rainbows",
            pendulum.date(2024, 4, 15),
            pendulum.date(2024, 4, 19),
        ),
    ],
    "Kokuu": [
        Kou(
            "葭始生",
            "Ashi hajimete shōzu",
            "First reeds sprout",
            pendulum.date(2024, 4, 20),
            pendulum.date(2024, 4, 24),
        ),
        Kou(
            "霜止出苗",
            "Shimo yamite nae izuru",
            "Last frost, rice seedlings grow",
            pendulum.date(2024, 4, 25),
            pendulum.date(2024, 4, 29),
        ),
        Kou(
            "牡丹華",
            "Botan hana saku",
            "Peonies bloom",
            pendulum.date(2024, 4, 30),
            pendulum.date(2024, 5, 4),
        ),
    ],
    "Rikka": [
        Kou(
            "蛙始鳴",
            "Kawazu hajimete naku",
            "Frogs start singing",
            pendulum.date(2024, 5, 5),
            pendulum.date(2024, 5, 9),
        ),
        Kou(
            "蚯蚓出",
            "Mimizu izuru",
            "Worms surface",
            pendulum.date(2024, 5, 10),
            pendulum.date(2024, 5, 14),
        ),
        Kou(
            "竹笋生",
            "Takenoko shōzu",
            "Bamboo shoots sprout",
            pendulum.date(2024, 5, 15),
            pendulum.date(2024, 5, 20),
        ),
    ],
    "Shōman": [
        Kou(
            "蚕起食桑",
            "Kaiko okite kuwa o hamu",
            "Silkworms start feasting on mulberry leaves",
            pendulum.date(2024, 5, 21),
            pendulum.date(2024, 5, 25),
        ),
        Kou(
            "紅花栄",
            "Benibana sakau",
            "Safflowers bloom",
            pendulum.date(2024, 5, 26),
            pendulum.date(2024, 5, 30),
        ),
        Kou(
            "麦秋至",
            "Mugi no toki itaru",
            "Wheat ripens and is harvested",
            pendulum.date(2024, 5, 31),
            pendulum.date(2024, 6, 5),
        ),
    ],
    "Bōshu": [
        Kou(
            "蟷螂生",
            "Kamakiri shōzu",
            "Praying mantises hatch",
            pendulum.date(2024, 6, 6),
            pendulum.date(2024, 6, 10),
        ),
        Kou(
            "腐草為螢",
            "Kusaretaru kusa hotaru to naru",
            "Rotten grass becomes fireflies",
            pendulum.date(2024, 6, 11),
            pendulum.date(2024, 6, 15),
        ),
        Kou(
            "梅子黄",
            "Ume no mi kibamu",
            "Plums turn yellow",
            pendulum.date(2024, 6, 16),
            pendulum.date(2024, 6, 20),
        ),
    ],
    "Geshi": [
        Kou(
            "乃東枯",
            "Natsukarekusa karuru",
            "Self-heal withers",
            pendulum.date(2024, 6, 21),
            pendulum.date(2024, 6, 26),
        ),
        Kou(
            "菖蒲華",
            "Ayame hana saku",
            "Irises bloom",
            pendulum.date(2024, 6, 27),
            pendulum.date(2024, 7, 1),
        ),
        Kou(
            "半夏生",
            "Hange shōzu",
            "Crow-dipper sprouts",
            pendulum.date(2024, 7, 2),
            pendulum.date(2024, 7, 6),
        ),
    ],
    "Shōsho": [
        Kou(
            "温風至",
            "Atsukaze itaru",
            "Warm winds blow",
            pendulum.date(2024, 7, 7),
            pendulum.date(2024, 7, 11),
        ),
        Kou(
            "蓮始開",
            "Hasu hajimete hiraku",
            "First lotus blossoms",
            pendulum.date(2024, 7, 12),
            pendulum.date(2024, 7, 16),
        ),
        Kou(
            "鷹乃学習",
            "Taka sunawachi waza o narau",
            "Hawks learn to fly",
            pendulum.date(2024, 7, 17),
            pendulum.date(2024, 7, 22),
        ),
    ],
    "Taisho": [
        Kou(
            "桐始結花",
            "Kiri hajimete hana o musubu",
            "Paulownia trees produce seeds",
            pendulum.date(2024, 7, 23),
            pendulum.date(2024, 7, 28),
        ),
        Kou(
            "土潤溽暑",
            "Tsuchi uruōte mushi atsushi",
            "Earth is damp, air is humid",
            pendulum.date(2024, 7, 29),
            pendulum.date(2024, 8, 2),
        ),
        Kou(
            "大雨時行",
            "Taiu tokidoki furu",
            "Great rains sometimes fall",
            pendulum.date(2024, 8, 3),
            pendulum.date(2024, 8, 7),
        ),
    ],
    "Risshū": [
        Kou(
            "涼風至",
            "Suzukaze itaru",
            "Cool winds blow",
            pendulum.date(2024, 8, 8),
            pendulum.date(2024, 8, 12),
        ),
        Kou(
            "寒蝉鳴",
            "Higurashi naku",
            "Evening cicadas sing",
            pendulum.date(2024, 8, 13),
            pendulum.date(2024, 8, 17),
        ),
        Kou(
            "蒙霧升降",
            "Fukaki kiri matō",
            "Thick fog descends",
            pendulum.date(2024, 8, 18),
            pendulum.date(2024, 8, 22),
        ),
    ],
    "Shosho": [
        Kou(
            "綿柎開",
            "Wata no hana shibe hiraku",
            "Cotton flowers bloom",
            pendulum.date(2024, 8, 23),
            pendulum.date(2024, 8, 27),
        ),
        Kou(
            "天地始粛",
            "Tenchi hajimete samushi",
            "Heat starts to die down",
            pendulum.date(2024, 8, 28),
            pendulum.date(2024, 9, 1),
        ),
        Kou(
            "禾乃登",
            "Kokumono sunawachi minoru",
            "Rice ripens",
            pendulum.date(2024, 9, 2),
            pendulum.date(2024, 9, 7),
        ),
    ],
    "Hakuro": [
        Kou(
            "草露白",
            "Kusa no tsuyu shiroshi",
            "Dew glistens white on grass",
            pendulum.date(2024, 9, 8),
            pendulum.date(2024, 9, 12),
        ),
        Kou(
            "鶺鴒鳴",
            "Sekirei naku",
            "Wagtails sing",
            pendulum.date(2024, 9, 13),
            pendulum.date(2024, 9, 17),
        ),
        Kou(
            "玄鳥去",
            "Tsubame saru",
            "Swallows leave",
            pendulum.date(2024, 9, 18),
            pendulum.date(2024, 9, 22),
        ),
    ],
    "Shūbun": [
        Kou(
            "雷乃収声",
            "Kaminari sunawachi koe o osamu",
            "Thunder ceases",
            pendulum.date(2024, 9, 23),
            pendulum.date(2024, 9, 27),
        ),
        Kou(
            "蟄虫坏戸",
            "Mushi kakurete to o fusagu",
            "Insects hole up underground",
            pendulum.date(2024, 9, 28),
            pendulum.date(2024, 10, 2),
        ),
        Kou(
            "水始涸",
            "Mizu hajimete karuru",
            "Farmers drain fields",
            pendulum.date(2024, 10, 3),
            pendulum.date(2024, 10, 7),
        ),
    ],
    "Kanro": [
        Kou(
            "鴻雁来",
            "Kōgan kitaru",
            "Wild geese return",
            pendulum.date(2024, 10, 8),
            pendulum.date(2024, 10, 12),
        ),
        Kou(
            "菊花開",
            "Kiku no hana hiraku",
            "Chrysanthemums bloom",
            pendulum.date(2024, 10, 13),
            pendulum.date(2024, 10, 17),
        ),
        Kou(
            "蟋蟀在戸",
            "Kirigirisu to ni ari",
            "Crickets chirp around the door",
            pendulum.date(2024, 10, 18),
            pendulum.date(2024, 10, 22),
        ),
    ],
    "Sōkō": [
        Kou(
            "霜始降",
            "Shimo hajimete furu",
            "First frost",
            pendulum.date(2024, 10, 23),
            pendulum.date(2024, 10, 27),
        ),
        Kou(
            "霎時施",
            "Kosame tokidoki furu",
            "Light rains sometimes fall",
            pendulum.date(2024, 10, 28),
            pendulum.date(2024, 11, 1),
        ),
        Kou(
            "楓蔦黄",
            "Momiji tsuta kibamu",
            "Maple leaves and ivy turn yellow",
            pendulum.date(2024, 11, 2),
            pendulum.date(2024, 11, 6),
        ),
    ],
    "Rittō": [
        Kou(
            "山茶始開",
            "Tsubaki hajimete hiraku",
            "Camellias bloom",
            pendulum.date(2024, 11, 7),
            pendulum.date(2024, 11, 11),
        ),
        Kou(
            "地始凍",
            "Chi hajimete kōru",
            "Land starts to freeze",
            pendulum.date(2024, 11, 12),
            pendulum.date(2024, 11, 16),
        ),
        Kou(
            "金盞香",
            "Kinsenka saku",
            "Daffodils bloom",
            pendulum.date(2024, 11, 17),
            pendulum.date(2024, 11, 21),
        ),
    ],
    "Shōsetsu": [
        Kou(
            "虹蔵不見",
            "Niji kakurete miezu",
            "Rainbows hide",
            pendulum.date(2024, 11, 22),
            pendulum.date(2024, 11, 26),
        ),
        Kou(
            "朔風払葉",
            "Kitakaze konoha o harau",
            "North wind blows the leaves from the trees",
            pendulum.date(2024, 11, 27),
            pendulum.date(2024, 12, 1),
        ),
        Kou(
            "橘始黄",
            "Tachibana hajimete kibamu",
            "Tachibana citrus tree leaves start to turn yellow",
            pendulum.date(2024, 12, 2),
            pendulum.date(2024, 12, 6),
        ),
    ],
    "Taisetsu": [
        Kou(
            "閉塞成冬",
            "Sora samuku fuyu to naru",
            "Cold sets in, winter begins",
            pendulum.date(2024, 12, 7),
            pendulum.date(2024, 12, 11),
        ),
        Kou(
            "熊蟄穴",
            "Kuma ana ni komoru",
            "Bears start hibernating in their dens",
            pendulum.date(2024, 12, 12),
            pendulum.date(2024, 12, 16),
        ),
        Kou(
            "鱖魚群",
            "Sake no uo muragaru",
            "Salmon gather and swim upstream",
            pendulum.date(2024, 12, 17),
            pendulum.date(2024, 12, 21),
        ),
    ],
    "Tōji": [
        Kou(
            "乃東生",
            "Natsukarekusa shōzu",
            "Self-heal sprouts",
            pendulum.date(2024, 12, 22),
            pendulum.date(2024, 12, 26),
        ),
        Kou(
            "麋角解",
            "Sawashika no tsuno otsuru",
            "Deer shed antlers",
            pendulum.date(2024, 12, 27),
            pendulum.date(2024, 12, 31),
        ),
        Kou(
            "雪下出麦",
            "Yuki watarite mugi nobiru",
            "Wheat sprouts under snow",
            pendulum.date(2024, 1, 1),
            pendulum.date(2024, 1, 4),
        ),
    ],
    "Shōkan": [
        Kou(
            "芹乃栄",
            "Seri sunawachi sakau",
            "Parsley flourishes",
            pendulum.date(2024, 1, 5),
            pendulum.date(2024, 1, 9),
        ),
        Kou(
            "水泉動",
            "Shimizu atataka o fukumu",
            "Springs thaw",
            pendulum.date(2024, 1, 10),
            pendulum.date(2024, 1, 14),
        ),
        Kou(
            "雉始雊",
            "Kiji hajimete naku",
            "Pheasants start to call",
            pendulum.date(2024, 1, 15),
            pendulum.date(2024, 1, 19),
        ),
    ],
    "Daikan": [
        Kou(
            "款冬華",
            "Fuki no hana saku",
            "Butterburs bud",
            pendulum.date(2024, 1, 20),
            pendulum.date(2024, 1, 24),
        ),
        Kou(
            "水沢腹堅",
            "Sawamizu kōri tsumeru",
            "Ice thickens on streams",
            pendulum.date(2024, 1, 25),
            pendulum.date(2024, 1, 29),
        ),
        Kou(
            "鶏始乳",
            "Niwatori hajimete toya ni tsuku",
            "Hens start laying eggs",
            pendulum.date(2024, 1, 30),
            pendulum.date(2024, 2, 3),
        ),
    ],
}

# Create a list of Term (Season24) objects
SEKKI_DICT = [
    Term("Risshun", "Beginning of spring", KOU_DICT["Risshun"]),
    Term("Usui", "Rainwater", KOU_DICT["Usui"]),
    Term("Keichitsu", "Insects awaken", KOU_DICT["Keichitsu"]),
    Term("Shunbun", "Spring equinox", KOU_DICT["Shunbun"]),
    Term("Seimei", "Pure and clear", KOU_DICT["Seimei"]),
    Term("Kokuu", "Grain rains", KOU_DICT["Kokuu"]),
    Term("Rikka", "Beginning of summer", KOU_DICT["Rikka"]),
    Term("Shōman", "Lesser ripening", KOU_DICT["Shōman"]),
    Term("Bōshu", "Grain beards and seeds", KOU_DICT["Bōshu"]),
    Term("Geshi", "Summer solstice", KOU_DICT["Geshi"]),
    Term("Shōsho", "Lesser heat", KOU_DICT["Shōsho"]),
    Term("Taisho", "Greater heat", KOU_DICT["Taisho"]),
    Term("Risshū", "Beginning of autumn", KOU_DICT["Risshū"]),
    Term("Shosho", "Manageable heat", KOU_DICT["Shosho"]),
    Term("Hakuro", "White dew", KOU_DICT["Hakuro"]),
    Term("Shūbun", "Autumn equinox", KOU_DICT["Shūbun"]),
    Term("Kanro", "Cold dew", KOU_DICT["Kanro"]),
    Term("Sōkō", "Frost falls", KOU_DICT["Sōkō"]),
    Term("Rittō", "Beginning of winter", KOU_DICT["Rittō"]),
    Term("Shōsetsu", "Lesser snow", KOU_DICT["Shōsetsu"]),
    Term("Taisetsu", "Greater snow", KOU_DICT["Taisetsu"]),
    Term("Tōji", "Winter solstice", KOU_DICT["Tōji"]),
    Term("Shōkan", "Lesser cold", KOU_DICT["Shōkan"]),
    Term("Daikan", "Greater cold", KOU_DICT["Daikan"]),
]


def get_kou(date):
    """Returns the Kou object for the given date."""
    for sekki in SEKKI_DICT:
        for kou in sekki.kous:
            if kou.start_date <= date <= kou.end_date:
                return kou
    return None


def get_term(date):
    """Determines the Term object for the provided date."""
    for sekki in SEKKI_DICT:
        if sekki.start_date <= date <= sekki.end_date:
            return sekki
    return None


# Example usage
today = pendulum.date.today()

cur_term = get_term(today)
if cur_term:
    print(f"# Current sekki (season24): {cur_term}")
    print(f"+++ Next sekki: {cur_term.get_next_term()}")
    print(f"--- Previous sekki: {cur_term.get_previous_term()}")
else:
    print("No sekki (season24) found for today's date.")


cur_kou = get_kou(today)

if cur_kou:
    print(f"### Current kou (micro-season): {cur_kou}")
    print(f"+++++ Next kou: {cur_kou.get_next_kou()}")
    print(f"----- Previous kou: {cur_kou.get_previous_kou()}")
else:
    print("No kou (micro-season) found for today's date.")

# GenAI
# Generate description with custom prompt
custom_prompt = "Focus on the natural phenomena occurring during this period."
gpt3_description = cur_kou.generate_description(
    llm_type="gpt3", custom_prompt=custom_prompt
)
if gpt3_description:
    print(f"Generated description (GPT-3): {gpt3_description}")
# Generate description with Claude
claude_description = cur_kou.generate_description(
    llm_type="claude", custom_prompt=custom_prompt
)
if claude_description:
    print(f"Generated description (Claude): {claude_description}")


# Generate image with custom prompt and save to file
custom_prompt = "featuring a sumo wrestler, a geisha in traditional attire, and a group of macaque monkeys nearby"

image = cur_kou.get_previous_kou().generate_image(
    ai_model="stable_diffusion",
    style="splatter paint",
    custom_prompt=custom_prompt,
)

if image:
    print("Image generated successfully")
