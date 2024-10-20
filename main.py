import datetime
import requests
from PIL import Image
import io
import os
import time
import openai
from openai import OpenAI
from anthropic import (
    Anthropic,
    HUMAN_PROMPT,
    AI_PROMPT,
    APIError,
    RateLimitError as AnthropicRateLimitError,
)
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client
import grpc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Function to get environment variable and raise ValueError if not set
def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} is not set in the .env file")
    return value


# Get API keys from .env file
openai_api_key = get_env_variable("OPENAI_API_KEY")
stability_api_key = get_env_variable("STABILITY_API_KEY")
anthropic_api_key = get_env_variable("ANTHROPIC_API_KEY")

# Initialize clients with API keys
openai_client = OpenAI(api_key=openai_api_key)
stability_api = client.StabilityInference(key=stability_api_key)
anthropic_client = Anthropic(api_key=anthropic_api_key)


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
        return [
            "Focus on the natural phenomena occurring during this period.",
            "Describe the agricultural activities associated with this season.",
            "Explain the cultural significance of this micro-season in Japan.",
            "Highlight the changes in flora and fauna during this time.",
            "Discuss any festivals or traditions linked to this micro-season.",
            "Describe the typical weather patterns during this period.",
            "Explain how this micro-season affects daily life in Japan.",
            "Highlight any poetry or literature associated with this season.",
            "Describe the seasonal foods and dishes popular during this time.",
            "Explain how this micro-season reflects Japanese aesthetics and philosophy.",
        ]

    def generate_image(
        self,
        ai_model="stable_diffusion",
        style="realistic",
        custom_prompt="",
        filename=None,
    ):
        """Generate image using specified AI model and style"""
        try:
            if ai_model == "stable_diffusion":
                image = self._generate_stable_diffusion_image(style, custom_prompt)
            elif ai_model == "dall_e":
                image = self._generate_dall_e_image(style, custom_prompt)
            else:
                raise ValueError(f"Unsupported AI model: {ai_model}")

            if image:
                self.images.append(image)
                if filename:
                    self._save_image(image, ai_model, filename)
                return image
            return None
        except (APIError, RateLimitError) as e:
            print(f"Error generating image: {str(e)}")
            return None

    def _generate_stable_diffusion_image(self, style, custom_prompt):
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                prompt = self._generate_image_prompt(style, custom_prompt)
                answers = stability_api.generate(prompt=prompt)
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            return Image.open(io.BytesIO(artifact.binary))
                return None
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise RateLimitError(
                            "Rate limit exceeded for Stable Diffusion API"
                        )
                else:
                    raise APIError(f"Stable Diffusion API error: {str(e)}")

    def _generate_dall_e_image(self, style, custom_prompt):
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                prompt = self._generate_image_prompt(style, custom_prompt)
                response = client.images.generate(prompt=prompt, n=1, size="1024x1024")
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
        base_prompt = f"An image representing the Japanese micro-season '{self.english}' ({self.kanji})"

        style_prompts = {
            "realistic": f"{base_prompt} in a realistic style",
            "anime": f"{base_prompt} in anime style",
            "ukiyo-e": f"{base_prompt} in the style of traditional Japanese ukiyo-e art",
            "watercolor": f"{base_prompt} as a watercolor painting",
            "digital art": f"{base_prompt} as digital art",
            "pencil sketch": f"{base_prompt} as a detailed pencil sketch",
        }

        full_prompt = (
            f"{style_prompts.get(style, base_prompt)}. {custom_prompt}".strip()
        )
        return full_prompt

    def _save_image(self, image, ai_model, filename):
        """Save the generated image with timestamp and model information"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        file_extension = (
            os.path.splitext(filename)[1]
            if filename.endswith((".png", ".jpg", ".jpeg"))
            else ".png"
        )
        full_filename = f"{filename}_{timestamp}_{ai_model}{file_extension}"
        image.save(full_filename)
        print(f"Image saved as: {full_filename}")

    @staticmethod
    def get_image_prompt_suggestions():
        """Return a list of prompt suggestions for image generation"""
        return [
            ## THIS CONTAINS TEMPORAL ASPECT which interferes with the image subject (base prompt == microseason)
            # "with cherry blossoms in the foreground",
            "featuring a traditional Japanese garden",
            "with Mount Fuji in the background",
            "showing a busy Tokyo street",
            "in a serene bamboo forest",
            "with koi fish in a pond",
            "featuring a majestic red torii gate",
            "with a traditional tea ceremony setting",
            "featuring iconic Japanese architecture",
            "with a bonsai tree in the foreground",
            "showing a peaceful Zen rock garden",
            "with a group of macaque monkeys nearby",
            "featuring a sumo wrestler",
            "with geisha in traditional attire",
            "showing a bullet train passing by",
            "with Studio Ghibli-inspired elements",
            "featuring Pikachu from Pokémon",
            "with Naruto performing a jutsu",
            "showing Sailor Moon and her team",
            "with Goku from Dragon Ball powering up",
            "featuring Totoro from My Neighbor Totoro",
            "with Attack on Titan's Eren Yeager",
            "showing One Piece's Monkey D. Luffy",
            "with Death Note's Light and L",
            "featuring characters from Demon Slayer",
            "with samurai armor on display",
            "showing a sushi chef at work",
            "with paper lanterns illuminating the scene",
        ]

    # OTHER DESCRIPTIVE METHODS
    def get_start_date(self):
        return self.start_date

    def get_end_date(self):
        return self.end_date

    def get_next_kou(self):
        for term in TERM_DICT:
            for i, kou in enumerate(term.kous):
                if kou == self:
                    if i < len(term.kous) - 1:
                        return term.kous[i + 1]
                    else:
                        next_term_index = (TERM_DICT.index(term) + 1) % len(TERM_DICT)
                        return TERM_DICT[next_term_index].kous[0]
        return None

    def get_previous_kou(self):
        for term in TERM_DICT:
            for i, kou in enumerate(term.kous):
                if kou == self:
                    if i > 0:
                        return term.kous[i - 1]
                    else:
                        prev_term_index = (TERM_DICT.index(term) - 1) % len(TERM_DICT)
                        return TERM_DICT[prev_term_index].kous[-1]
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
        current_index = TERM_DICT.index(self)
        return TERM_DICT[(current_index + 1) % len(TERM_DICT)]

    def get_previous_term(self):
        current_index = TERM_DICT.index(self)
        return TERM_DICT[(current_index - 1) % len(TERM_DICT)]

    def __str__(self):
        return f"{self.name} ({self.desc}) [{self.get_start_date()}...{self.get_end_date()}]"


# Create a list of Kou objects for each season
KOU_DICT = {
    "Risshun": [
        Kou(
            "東風解凍",
            "Harukaze kōri o toku",
            "East wind melts the ice",
            datetime.date(2024, 2, 4),
            datetime.date(2024, 2, 8),
        ),
        Kou(
            "黄鶯睍睆",
            "Kōō kenkan su",
            "Bush warblers start singing in the mountains",
            datetime.date(2024, 2, 9),
            datetime.date(2024, 2, 13),
        ),
        Kou(
            "魚上氷",
            "Uo kōri o izuru",
            "Fish emerge from the ice",
            datetime.date(2024, 2, 14),
            datetime.date(2024, 2, 18),
        ),
    ],
    "Usui": [
        Kou(
            "土脉潤起",
            "Tsuchi no shō uruoi okoru",
            "Rain moistens the soil",
            datetime.date(2024, 2, 19),
            datetime.date(2024, 2, 23),
        ),
        Kou(
            "霞始靆",
            "Kasumi hajimete tanabiku",
            "Mist starts to linger",
            datetime.date(2024, 2, 24),
            datetime.date(2024, 2, 28),
        ),
        Kou(
            "草木萌動",
            "Sōmoku mebae izuru",
            "Grass sprouts, trees bud",
            datetime.date(2024, 3, 1),
            datetime.date(2024, 3, 5),
        ),
    ],
    "Keichitsu": [
        Kou(
            "蟄虫啓戸",
            "Sugomori mushito o hiraku",
            "Hibernating insects surface",
            datetime.date(2024, 3, 6),
            datetime.date(2024, 3, 10),
        ),
        Kou(
            "桃始笑",
            "Momo hajimete saku",
            "First peach blossoms",
            datetime.date(2024, 3, 11),
            datetime.date(2024, 3, 15),
        ),
        Kou(
            "菜虫化蝶",
            "Namushi chō to naru",
            "Caterpillars become butterflies",
            datetime.date(2024, 3, 16),
            datetime.date(2024, 3, 20),
        ),
    ],
    "Shunbun": [
        Kou(
            "雀始巣",
            "Suzume hajimete sukū",
            "Sparrows start to nest",
            datetime.date(2024, 3, 21),
            datetime.date(2024, 3, 25),
        ),
        Kou(
            "櫻始開",
            "Sakura hajimete saku",
            "First cherry blossoms",
            datetime.date(2024, 3, 26),
            datetime.date(2024, 3, 30),
        ),
        Kou(
            "雷乃発声",
            "Kaminari sunawachi koe o hassu",
            "Distant thunder",
            datetime.date(2024, 3, 31),
            datetime.date(2024, 4, 4),
        ),
    ],
    "Seimei": [
        Kou(
            "玄鳥至",
            "Tsubame kitaru",
            "Swallows return",
            datetime.date(2024, 4, 5),
            datetime.date(2024, 4, 9),
        ),
        Kou(
            "鴻雁北",
            "Kōgan kaeru",
            "Wild geese fly north",
            datetime.date(2024, 4, 10),
            datetime.date(2024, 4, 14),
        ),
        Kou(
            "虹始見",
            "Niji hajimete arawaru",
            "First rainbows",
            datetime.date(2024, 4, 15),
            datetime.date(2024, 4, 19),
        ),
    ],
    "Kokuu": [
        Kou(
            "葭始生",
            "Ashi hajimete shōzu",
            "First reeds sprout",
            datetime.date(2024, 4, 20),
            datetime.date(2024, 4, 24),
        ),
        Kou(
            "霜止出苗",
            "Shimo yamite nae izuru",
            "Last frost, rice seedlings grow",
            datetime.date(2024, 4, 25),
            datetime.date(2024, 4, 29),
        ),
        Kou(
            "牡丹華",
            "Botan hana saku",
            "Peonies bloom",
            datetime.date(2024, 4, 30),
            datetime.date(2024, 5, 4),
        ),
    ],
    "Rikka": [
        Kou(
            "蛙始鳴",
            "Kawazu hajimete naku",
            "Frogs start singing",
            datetime.date(2024, 5, 5),
            datetime.date(2024, 5, 9),
        ),
        Kou(
            "蚯蚓出",
            "Mimizu izuru",
            "Worms surface",
            datetime.date(2024, 5, 10),
            datetime.date(2024, 5, 14),
        ),
        Kou(
            "竹笋生",
            "Takenoko shōzu",
            "Bamboo shoots sprout",
            datetime.date(2024, 5, 15),
            datetime.date(2024, 5, 20),
        ),
    ],
    "Shōman": [
        Kou(
            "蚕起食桑",
            "Kaiko okite kuwa o hamu",
            "Silkworms start feasting on mulberry leaves",
            datetime.date(2024, 5, 21),
            datetime.date(2024, 5, 25),
        ),
        Kou(
            "紅花栄",
            "Benibana sakau",
            "Safflowers bloom",
            datetime.date(2024, 5, 26),
            datetime.date(2024, 5, 30),
        ),
        Kou(
            "麦秋至",
            "Mugi no toki itaru",
            "Wheat ripens and is harvested",
            datetime.date(2024, 5, 31),
            datetime.date(2024, 6, 5),
        ),
    ],
    "Bōshu": [
        Kou(
            "蟷螂生",
            "Kamakiri shōzu",
            "Praying mantises hatch",
            datetime.date(2024, 6, 6),
            datetime.date(2024, 6, 10),
        ),
        Kou(
            "腐草為螢",
            "Kusaretaru kusa hotaru to naru",
            "Rotten grass becomes fireflies",
            datetime.date(2024, 6, 11),
            datetime.date(2024, 6, 15),
        ),
        Kou(
            "梅子黄",
            "Ume no mi kibamu",
            "Plums turn yellow",
            datetime.date(2024, 6, 16),
            datetime.date(2024, 6, 20),
        ),
    ],
    "Geshi": [
        Kou(
            "乃東枯",
            "Natsukarekusa karuru",
            "Self-heal withers",
            datetime.date(2024, 6, 21),
            datetime.date(2024, 6, 26),
        ),
        Kou(
            "菖蒲華",
            "Ayame hana saku",
            "Irises bloom",
            datetime.date(2024, 6, 27),
            datetime.date(2024, 7, 1),
        ),
        Kou(
            "半夏生",
            "Hange shōzu",
            "Crow-dipper sprouts",
            datetime.date(2024, 7, 2),
            datetime.date(2024, 7, 6),
        ),
    ],
    "Shōsho": [
        Kou(
            "温風至",
            "Atsukaze itaru",
            "Warm winds blow",
            datetime.date(2024, 7, 7),
            datetime.date(2024, 7, 11),
        ),
        Kou(
            "蓮始開",
            "Hasu hajimete hiraku",
            "First lotus blossoms",
            datetime.date(2024, 7, 12),
            datetime.date(2024, 7, 16),
        ),
        Kou(
            "鷹乃学習",
            "Taka sunawachi waza o narau",
            "Hawks learn to fly",
            datetime.date(2024, 7, 17),
            datetime.date(2024, 7, 22),
        ),
    ],
    "Taisho": [
        Kou(
            "桐始結花",
            "Kiri hajimete hana o musubu",
            "Paulownia trees produce seeds",
            datetime.date(2024, 7, 23),
            datetime.date(2024, 7, 28),
        ),
        Kou(
            "土潤溽暑",
            "Tsuchi uruōte mushi atsushi",
            "Earth is damp, air is humid",
            datetime.date(2024, 7, 29),
            datetime.date(2024, 8, 2),
        ),
        Kou(
            "大雨時行",
            "Taiu tokidoki furu",
            "Great rains sometimes fall",
            datetime.date(2024, 8, 3),
            datetime.date(2024, 8, 7),
        ),
    ],
    "Risshū": [
        Kou(
            "涼風至",
            "Suzukaze itaru",
            "Cool winds blow",
            datetime.date(2024, 8, 8),
            datetime.date(2024, 8, 12),
        ),
        Kou(
            "寒蝉鳴",
            "Higurashi naku",
            "Evening cicadas sing",
            datetime.date(2024, 8, 13),
            datetime.date(2024, 8, 17),
        ),
        Kou(
            "蒙霧升降",
            "Fukaki kiri matō",
            "Thick fog descends",
            datetime.date(2024, 8, 18),
            datetime.date(2024, 8, 22),
        ),
    ],
    "Shosho": [
        Kou(
            "綿柎開",
            "Wata no hana shibe hiraku",
            "Cotton flowers bloom",
            datetime.date(2024, 8, 23),
            datetime.date(2024, 8, 27),
        ),
        Kou(
            "天地始粛",
            "Tenchi hajimete samushi",
            "Heat starts to die down",
            datetime.date(2024, 8, 28),
            datetime.date(2024, 9, 1),
        ),
        Kou(
            "禾乃登",
            "Kokumono sunawachi minoru",
            "Rice ripens",
            datetime.date(2024, 9, 2),
            datetime.date(2024, 9, 7),
        ),
    ],
    "Hakuro": [
        Kou(
            "草露白",
            "Kusa no tsuyu shiroshi",
            "Dew glistens white on grass",
            datetime.date(2024, 9, 8),
            datetime.date(2024, 9, 12),
        ),
        Kou(
            "鶺鴒鳴",
            "Sekirei naku",
            "Wagtails sing",
            datetime.date(2024, 9, 13),
            datetime.date(2024, 9, 17),
        ),
        Kou(
            "玄鳥去",
            "Tsubame saru",
            "Swallows leave",
            datetime.date(2024, 9, 18),
            datetime.date(2024, 9, 22),
        ),
    ],
    "Shūbun": [
        Kou(
            "雷乃収声",
            "Kaminari sunawachi koe o osamu",
            "Thunder ceases",
            datetime.date(2024, 9, 23),
            datetime.date(2024, 9, 27),
        ),
        Kou(
            "蟄虫坏戸",
            "Mushi kakurete to o fusagu",
            "Insects hole up underground",
            datetime.date(2024, 9, 28),
            datetime.date(2024, 10, 2),
        ),
        Kou(
            "水始涸",
            "Mizu hajimete karuru",
            "Farmers drain fields",
            datetime.date(2024, 10, 3),
            datetime.date(2024, 10, 7),
        ),
    ],
    "Kanro": [
        Kou(
            "鴻雁来",
            "Kōgan kitaru",
            "Wild geese return",
            datetime.date(2024, 10, 8),
            datetime.date(2024, 10, 12),
        ),
        Kou(
            "菊花開",
            "Kiku no hana hiraku",
            "Chrysanthemums bloom",
            datetime.date(2024, 10, 13),
            datetime.date(2024, 10, 17),
        ),
        Kou(
            "蟋蟀在戸",
            "Kirigirisu to ni ari",
            "Crickets chirp around the door",
            datetime.date(2024, 10, 18),
            datetime.date(2024, 10, 22),
        ),
    ],
    "Sōkō": [
        Kou(
            "霜始降",
            "Shimo hajimete furu",
            "First frost",
            datetime.date(2024, 10, 23),
            datetime.date(2024, 10, 27),
        ),
        Kou(
            "霎時施",
            "Kosame tokidoki furu",
            "Light rains sometimes fall",
            datetime.date(2024, 10, 28),
            datetime.date(2024, 11, 1),
        ),
        Kou(
            "楓蔦黄",
            "Momiji tsuta kibamu",
            "Maple leaves and ivy turn yellow",
            datetime.date(2024, 11, 2),
            datetime.date(2024, 11, 6),
        ),
    ],
    "Rittō": [
        Kou(
            "山茶始開",
            "Tsubaki hajimete hiraku",
            "Camellias bloom",
            datetime.date(2024, 11, 7),
            datetime.date(2024, 11, 11),
        ),
        Kou(
            "地始凍",
            "Chi hajimete kōru",
            "Land starts to freeze",
            datetime.date(2024, 11, 12),
            datetime.date(2024, 11, 16),
        ),
        Kou(
            "金盞香",
            "Kinsenka saku",
            "Daffodils bloom",
            datetime.date(2024, 11, 17),
            datetime.date(2024, 11, 21),
        ),
    ],
    "Shōsetsu": [
        Kou(
            "虹蔵不見",
            "Niji kakurete miezu",
            "Rainbows hide",
            datetime.date(2024, 11, 22),
            datetime.date(2024, 11, 26),
        ),
        Kou(
            "朔風払葉",
            "Kitakaze konoha o harau",
            "North wind blows the leaves from the trees",
            datetime.date(2024, 11, 27),
            datetime.date(2024, 12, 1),
        ),
        Kou(
            "橘始黄",
            "Tachibana hajimete kibamu",
            "Tachibana citrus tree leaves start to turn yellow",
            datetime.date(2024, 12, 2),
            datetime.date(2024, 12, 6),
        ),
    ],
    "Taisetsu": [
        Kou(
            "閉塞成冬",
            "Sora samuku fuyu to naru",
            "Cold sets in, winter begins",
            datetime.date(2024, 12, 7),
            datetime.date(2024, 12, 11),
        ),
        Kou(
            "熊蟄穴",
            "Kuma ana ni komoru",
            "Bears start hibernating in their dens",
            datetime.date(2024, 12, 12),
            datetime.date(2024, 12, 16),
        ),
        Kou(
            "鱖魚群",
            "Sake no uo muragaru",
            "Salmon gather and swim upstream",
            datetime.date(2024, 12, 17),
            datetime.date(2024, 12, 21),
        ),
    ],
    "Tōji": [
        Kou(
            "乃東生",
            "Natsukarekusa shōzu",
            "Self-heal sprouts",
            datetime.date(2024, 12, 22),
            datetime.date(2024, 12, 26),
        ),
        Kou(
            "麋角解",
            "Sawashika no tsuno otsuru",
            "Deer shed antlers",
            datetime.date(2024, 12, 27),
            datetime.date(2024, 12, 31),
        ),
        Kou(
            "雪下出麦",
            "Yuki watarite mugi nobiru",
            "Wheat sprouts under snow",
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 4),
        ),
    ],
    "Shōkan": [
        Kou(
            "芹乃栄",
            "Seri sunawachi sakau",
            "Parsley flourishes",
            datetime.date(2024, 1, 5),
            datetime.date(2024, 1, 9),
        ),
        Kou(
            "水泉動",
            "Shimizu atataka o fukumu",
            "Springs thaw",
            datetime.date(2024, 1, 10),
            datetime.date(2024, 1, 14),
        ),
        Kou(
            "雉始雊",
            "Kiji hajimete naku",
            "Pheasants start to call",
            datetime.date(2024, 1, 15),
            datetime.date(2024, 1, 19),
        ),
    ],
    "Daikan": [
        Kou(
            "款冬華",
            "Fuki no hana saku",
            "Butterburs bud",
            datetime.date(2024, 1, 20),
            datetime.date(2024, 1, 24),
        ),
        Kou(
            "水沢腹堅",
            "Sawamizu kōri tsumeru",
            "Ice thickens on streams",
            datetime.date(2024, 1, 25),
            datetime.date(2024, 1, 29),
        ),
        Kou(
            "鶏始乳",
            "Niwatori hajimete toya ni tsuku",
            "Hens start laying eggs",
            datetime.date(2024, 1, 30),
            datetime.date(2024, 2, 3),
        ),
    ],
}

# Create a list of Term (Season24) objects
TERM_DICT = [
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
    for term in TERM_DICT:
        for kou in term.kous:
            if kou.start_date <= date <= kou.end_date:
                return kou
    return None


def get_term(date):
    """Determines the Term object for the provided date."""
    for term in TERM_DICT:
        if term.start_date <= date <= term.end_date:
            return term
    return None


# Example usage
today = datetime.date.today()

cur_term = get_term(today)
if cur_term:
    print(f"# Current term (season24): {cur_term}")
    print(f"+++ Next term: {cur_term.get_next_term()}")
    print(f"--- Previous term: {cur_term.get_previous_term()}")
else:
    print("No term (season24) found for today's date.")


cur_kou = get_kou(today)

if cur_kou:
    print(f"### Current kou (micro-season): {cur_kou}")
    print(f"+++++ Next kou: {cur_kou.get_next_kou()}")
    print(f"----- Previous kou: {cur_kou.get_previous_kou()}")
else:
    print("No kou (micro-season) found for today's date.")

# GenAI
print("Description prompt suggestions:")
for suggestion in cur_kou.get_description_prompt_suggestions():
    print(f"- {suggestion}")
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


print("Image prompt suggestions:")
for suggestion in cur_kou.get_image_prompt_suggestions():
    print(f"- {suggestion}")

# Generate image with custom prompt and save to file
custom_prompt = "with Totoro from My Neighbor Totoro sitting under a red torii gate"
image = cur_kou.generate_image(
    ai_model="stable_diffusion",
    style="ukiyo-e",
    custom_prompt=custom_prompt,
    filename="east_wind_image",
)

if image:
    print("Image generated successfully")
