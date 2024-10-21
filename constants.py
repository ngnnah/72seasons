DESCRIPTION_SUGGESTIONS = [
    "Focus on the natural phenomena occurring during this period",
    "Describe the agricultural activities associated with this season",
    "Explain the cultural significance of this micro-season in Japan",
    "Highlight the changes in flora and fauna during this time",
    "Discuss any festivals or traditions linked to this micro-season",
    "Describe the typical weather patterns during this period",
    "Explain how this micro-season affects daily life in Japan",
    "Highlight any poetry or literature associated with this season",
    "Describe the seasonal foods and dishes popular during this time",
    "Explain how this micro-season reflects Japanese aesthetics and philosophy",
]
IMAGE_SUGGESTIONS = [
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
STYLE_PRESETS = {
    ## PHOTO STYLES ##
    "indoor": "Image captured inside a structure",
    "outdoor": "Image inspired by natural settings",
    "bokeh": "Image with a blurred background, emphasizing the subject",
    "black and white": "Monochromatic image focused on contrast and texture",
    "close up": "Image tightly framed, highlighting details and textures",
    "advertising - testimonial": "Image incorporating customer reviews to build trust",
    "cyberpunk": "Futuristic image featuring neon lights and technology",
    "fish eye": "Image with a distinctive ultra-wide perspective",
    "studio shot": "Image professionally lit and composed, emphasizing the subject",
    "wet plate": "Image with a vintage texture and antique feel",
    "motion blur": "Image capturing a sense of speed or action",
    "instant photo": "Image resembling retro instant film prints",
    "moody portrait": "Image with dramatic lighting, darker tones, and emotions",
    "bold portrait": "Image emphasizing personality using strong lighting and colors",
    "water action": "Image capturing movement in water-based activities",
    "sunset": "Image featuring the warm hues of a setting sun",
    "brutalist product": "Image highlighting form and function using bold shapes and raw materials",
    "summer product": "Image evoking warmth and leisure by showcasing products in sunny settings",
    "advertising - product": "Image showcasing a product's features with clean, well-lit compositions",
    "nautical product": "Image incorporating water, boats, and coastal themes",
    "advertising - podium": "Image displaying products on a pedestal for prominence",
    "framed": "Image with a visible border, resembling a framed picture",
    "hieroglyph": "Image inspired by ancient Egyptian hieroglyphs",
    ## ART STYLES ##
    "watercolor": "Image resembling traditional watercolor paintings",
    "pastel drawing": "Image with soft colors and subtle shading",
    "cartoon": "Playful illustration with exaggerated features and bright colors",
    "oil painting": "Image mimicking traditional oil artwork",
    "pencil sketch": "Image resembling hand-drawn sketches",
    "paper collage": "Image incorporating textures and patterns from paper",
    "street art": "Image inspired by graffiti and urban culture",
    "psychedelic": "Image featuring vibrant colors, swirling patterns, and surreal imagery",
    "ukiyo-e": "Image inspired by Japanese woodblock prints depicting landscapes and nature",
    "baroque": "Image featuring ornate details, rich colors, and grandeur",
    "coloring book": "Simplified line art for coloring",
    "pop art": "Image using bold colors, contrasts, and pop culture references",
    "art nouveau": "Image incorporating organic shapes, elegant lines, and intricate details",
    "anime": "Image inspired by Japanese animation style",
    "manga": "Image inspired by Japanese comic books",
    "anthropomorphic": "Image featuring animals or objects with human-like traits",
    "vintage boho": "Image blending vintage and bohemian elements",
    "floral": "Image featuring flowers and botanical elements",
    "cubism": "Image using geometric shapes and fragmented forms",
    "tile art": "Image resembling decorative tiles",
    "medieval": "Image inspired by medieval art and architecture",
    "surreal": "Image capturing dreamlike, fantastical, or illogical elements",
    "rock album": "Image designed for music albums",
    "kids illustration": "Playful and colorful image for children",
    "steampunk portrait": "Image featuring a steampunk aesthetic",
    "expressionist": "Image evoking emotion and atmosphere",
    "creepy": "Image with a dark, unsettling atmosphere",
    "fantasy cartoon": "Image featuring fantastical characters and settings",
    "mosaic": "Image composed of small, colored pieces",
    ## DIGITAL STYLES ##
    "pixel art": "Image resembling early video game graphics",
    "blog illustration": "Image designed to accompany blog posts",
    "app icon": "Small, simple graphic representing mobile apps",
    "cute sticker": "Small, playful illustration",
    "flat design": "Image using bold colors, simple shapes, and minimal detail",
    "vector vignette": "Image featuring a central image with a decorative border",
    "line sticker": "Simple illustration created using continuous lines",
    "papercraft": "Image inspired by the creation of 3D objects from paper",
    "futuristic": "Image depicting advanced technology and future landscapes",
    "splatter paint": "Image featuring paint splatters and drips",
    "minimal line art": "Image featuring simple, continuous lines",
    "emoji": "Small graphic conveying emotions",
    "tiki": "Image inspired by Polynesian culture and design",
    "stained glass": "Image resembling stained glass windows",
    "tropical": "Image featuring lush, tropical scenes",
    "punk": "Image featuring elements of punk culture",
    "neon avatar": "Character design featuring bright neon colors",
    "sports logo": "Design representing athletic teams or events",
    ## 3D STYLES ##
    "cute isometric": "Illustration using the isometric perspective",
    "neon": "Image featuring bright, glowing colors",
    "low poly": "Image using geometric shapes and a limited number of polygons",
    "metallic": "Image showcasing reflective, shiny, or textured qualities of metal",
    "cute character": "Illustration featuring endearing characters",
    "cute avatar": "Digital character design with an adorable aesthetic",
    "origami": "Image inspired by Japanese paper folding",
    "mecha": "Illustration of robotic, mechanical characters or vehicles",
    "animated character": "Illustration resembling cartoons",
    "clay model": "Image with a textured, handcrafted feel",
    "wood carving": "Image inspired by the look and feel of wood carvings",
    "minimal balloon": "Image featuring minimalistic balloon-like shapes",
    "stone carving": "Image with rough textures resembling stone carvings",
    "plastic bricks": "Image inspired by the look of toy building bricks",
    "fluffy": "Image with soft textures",
    "multicolor mecha": "Illustration of robots with vibrant, multicolored designs",
    "mythical": "Image featuring fantastical creatures and settings",
    ## SCENE STYLES ##
    "miniature diorama": "Image resembling small-scale, 3D scenes",
    "future tokyo": "Image depicting futuristic visions of Tokyo",
    "city fashion": "Image featuring characters and clothing in urban settings",
    "winter scene": "Image featuring snow, ice, and a chilly atmosphere",
    "country living": "Image depicting rural settings with a focus on nature",
    "city kitchen": "Image featuring modern, urban kitchen designs",
    "plush bedroom": "Image featuring opulent, comfortable bedrooms",
    "supermarket": "Image featuring a grocery store setting",
    "beachside luxury house": "Image showcasing opulent seaside homes",
    "colorful living": "Image featuring vibrant, colorful living spaces",
    "wheat field": "Image showcasing expansive plots of agricultural land",
}
