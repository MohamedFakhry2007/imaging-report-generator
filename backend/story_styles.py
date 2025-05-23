# backend/story_styles.py

# Each style is a dictionary with 'id', 'name' (for UI, in Arabic), and 'prompt' (for the AI)
STORY_STYLES = [
    {
        "id": "general_modern_standard",
        "name": "عربي فصيح حديث (افتراضي)",
        "prompt": "اكتب بأسلوب عربي فصيح حديث وواضح. يجب أن تكون القصة سهلة الفهم وجذابة لجمهور واسع. تجنب التعقيد اللفظي المفرط وركز على سلاسة السرد وجماليات اللغة البسيطة والمعبرة."
    },
    {
        "id": "classical_poetic",
        "name": "فصيح تراثي وشعري",
        "prompt": "اكتب بأسلوب لغوي تراثي، مستلهمًا جماليات النثر العربي القديم. استخدم مفردات غنية وبناء جمل فيه جزالة وفخامة. يمكن أن يتضمن السرد بعض الصور الشعرية والاستعارات البلاغية. اجعل القصة تبدو وكأنها قطعة من أدب العصور الذهبية."
    },
    {
        "id": "simple_children",
        "name": "مبسط للأطفال",
        "prompt": "اكتب قصة مناسبة للأطفال، باستخدام لغة بسيطة جداً وجمل قصيرة ومفهومة. يجب أن تكون القصة مسلية وتحمل قيماً إيجابية. ركز على الحوارات الواضحة والأحداث المشوقة. تجنب الكلمات الصعبة أو المفاهيم المجردة."
    },
    {
        "id": "suspense_mystery",
        "name": "تشويق وغموض",
        "prompt": "اكتب بأسلوب يركز على التشويق والغموض. استخدم بناءاً سردياً يزيد من التوتر تدريجياً، مع تلميحات وإشارات مبهمة تثير فضول القارئ. يجب أن تكون النهاية مفاجئة أو تكشف سراً غير متوقع. اللغة يجب أن تكون دقيقة وموحية."
    },
    {
        "id": "humorous_sarcastic",
        "name": "فكاهي وساخر",
        "prompt": "اكتب بأسلوب فكاهي وساخر. استخدم المفارقات اللفظية والمواقف المضحكة. يمكن أن يكون السرد ناقداً بطريقة غير مباشرة. اللغة يجب أن تكون حيوية ومليئة بالذكاء اللفظي."
    }
]

# Example of how to load the arabic_style_sample.txt for the default style if needed,
# or it can be directly embedded in the 'prompt' above.
# For now, the default style prompt is self-contained.
# try:
#     from pathlib import Path
#     ROOT_DIR = Path(__file__).parent
#     with open(ROOT_DIR / 'arabic_style_sample.txt', 'r', encoding='utf-8') as file:
#         GENERAL_MODERN_STANDARD_PROMPT = file.read()
#     # Update the default style prompt if you want to use the text file
#     # STORY_STYLES[0]["prompt"] = GENERAL_MODERN_STANDARD_PROMPT 
# except Exception as e:
#     print(f"Error loading arabic_style_sample.txt for default style: {e}")
#     # Fallback if file is not found, though it's better to ensure prompts are self-contained or files exist
#     if STORY_STYLES[0]["id"] == "general_modern_standard" and not STORY_STYLES[0]["prompt"]:
#         STORY_STYLES[0]["prompt"] = "فشل في تحميل النص المرجعي للأسلوب الافتراضي. اكتب بأسلوب عربي فصيح حديث."
