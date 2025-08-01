import gradio as gr
import json
import openai
import ollama
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI API (opcional)
openai.api_key = os.getenv("OPENAI_API_KEY")


class Veo3PromptGenerator:
    def __init__(self):
        # Verificar modelos disponibles en Ollama
        self.ollama_available = self.check_ollama()
        self.ollama_models = self.get_ollama_models() if self.ollama_available else []

        # Definir opciones predefinidas basadas en el tutorial
        self.shot_types = {
            "Close-up": "close-up",
            "Medium Shot": "medium shot", 
            "Wide Shot": "wide shot",
            "Extreme Close-up": "extreme close-up",
            "Over the Shoulder": "over the shoulder",
            "Two Shot": "two shot",
            "Establishing Shot": "establishing shot"
        }
        self.camera_movements = {
            "Estático": "static",
            "Dolly In": "slow dolly in",
            "Dolly Out": "slow dolly out",
            "Pan Izquierda": "pan left",
            "Pan Derecha": "pan right",
            "Tracking": "tracking shot",
            "Crane Up": "crane up",
            "Crane Down": "crane down",
            "Handheld": "handheld",
            "Steadicam": "steadicam following"
        }
        self.styles = {
            "Cinematográfico": "cinematic",
            "Documental": "documentary",
            "Anime": "anime",
            "Film Noir": "film noir",
            "Editorial": "editorial",
            "Fantasía": "cinematic fantasy",
            "Sci-Fi": "sci-fi futuristic",
            "Horror": "horror suspense",
            "Romántico": "romantic soft",
            "Deportivo": "sports dynamic"
        }
        self.lighting_types = {
            "Luz Natural Suave": "soft diffused natural light",
            "Hora Dorada": "golden hour",
            "Hora Azul": "blue hour",
            "Chiaroscuro": "chiaroscuro",
            "Neón": "neon moody high contrast",
            "Estudio": "clean studio lighting",
            "Contraluz": "backlit silhouette",
            "Velas": "candlelit warm",
            "Luna": "moonlight cold"
        }
        self.tones = {
            "Misterioso": "mysterious and suspenseful",
            "Alegre": "joyful and uplifting",
            "Tenso": "tense and dramatic",
            "Melancólico": "melancholic and introspective",
            "Épico": "epic and grandiose",
            "Cómico": "comedic and lighthearted",
            "Romántico": "romantic and tender",
            "Oscuro": "dark and brooding",
            "Esperanzador": "hopeful and inspiring"
        }
        self.times_of_day = {
            "Amanecer": "dawn",
            "Mañana": "morning",
            "Mediodía": "noon",
            "Atardecer": "dusk",
            "Noche": "night",
            "Madrugada": "late night",
            "Hora Mágica": "magic hour"
        }

        # Prompts creativos para Ollama
        self.creative_prompts = [
            "Una escena surrealista con elementos flotantes",
            "Un momento íntimo entre dos personajes",
            "Una persecución épica en un entorno único",
            "Un objeto cotidiano transformándose mágicamente",
            "Una revelación dramática con giro inesperado",
            "Un paisaje imposible que desafía la física",
            "Un encuentro entre lo antiguo y lo futurista",
            "Una danza de elementos naturales",
            "Un momento de soledad profunda",
            "Una celebración cultural vibrante"
        ]

    def check_ollama(self) -> bool:
        """Verifica si Ollama está instalado y funcionando"""
        try:
            ollama.list()
            return True
        except Exception as e:
            print("❌ Error al conectar con Ollama:", e)
            return False

    def get_ollama_models(self) -> List[str]:
        """Obtiene la lista de modelos disponibles en Ollama"""
        try:
            result = ollama.list()
            print("DEBUG: Respuesta completa de ollama.list():", result)  # Para diagnóstico
            
            if "models" in result:
                # CORRECCIÓN: Ahora usamos "model" en lugar de "name"
                models = [model.get("model", model.get("name", "")).strip() for model in result["models"]]
                return [name for name in models if name]  # Filtrar vacíos
            else:
                print("⚠️ Formato inesperado en respuesta de ollama.list():", result)
                return []
        except Exception as e:
            print("❌ Error al obtener modelos de Ollama:", e)
            return []

    def generate_ideas_with_ollama(self, 
                                 base_prompt: str, 
                                 model: str = "llama2",
                                 num_ideas: int = 5) -> List[str]:
        """Genera ideas creativas usando Ollama"""
        if not self.ollama_available:
            return ["Ollama no está disponible. Por favor, instálalo primero."]
        if not model or model == "No disponible":
            return ["❌ Por favor, selecciona un modelo válido."]
        
        # CORRECCIÓN: Usar solo el nombre base del modelo (sin tag)
        model_name = model.split(':')[0]
        
        try:
            prompt = f"""Genera {num_ideas} ideas creativas y cinematográficas para videos basadas en este concepto: "{base_prompt}"
            Para cada idea, incluye:
            - Una descripción visual única
            - Un elemento emocional o narrativo
            - Un detalle cinematográfico específico (ángulo, movimiento, iluminación)
            Formato: Lista numerada con descripciones concisas pero evocativas."""
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            ideas_text = response['message']['content']
            # Dividir las ideas en una lista
            ideas = [idea.strip() for idea in ideas_text.split('\n') if idea.strip() and any(char.isdigit() for char in idea[:3])]
            return ideas[:num_ideas] if ideas else ["No se pudieron generar ideas. Intenta con otro modelo."]
        except Exception as e:
            return [f"Error generando ideas: {str(e)}"]

    def enhance_prompt_with_ollama(self, 
                                 simple_prompt: str,
                                 style: str,
                                 tone: str,
                                 model: str = "llama2") -> Dict:
        """Mejora un prompt simple usando Ollama"""
        if not self.ollama_available:
            return {"error": "Ollama no está disponible"}
        if not model or model == "No disponible":
            return {"error": "Modelo no válido"}
        
        # CORRECCIÓN: Usar solo el nombre base del modelo (sin tag)
        model_name = model.split(':')[0]
        
        try:
            prompt = f"""Como experto en cinematografía y dirección de videos, mejora esta descripción para un video:
Descripción original: "{simple_prompt}"
Estilo deseado: {style}
Tono emocional: {tone}
Proporciona una descripción mejorada que incluya:
1. Descripción detallada del sujeto principal
2. Acción específica y dinámica
3. Detalles del entorno y atmósfera
4. Sugerencias de iluminación cinematográfica
5. Elementos visuales únicos o efectos
6. Sugerencias de audio y música
Responde en formato estructurado para fácil procesamiento."""
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            enhanced_text = response['message']['content']
            # Procesar la respuesta para extraer elementos
            enhanced_elements = self.parse_ollama_response(enhanced_text, simple_prompt)
            return enhanced_elements
        except Exception as e:
            return {"error": f"Error mejorando prompt: {str(e)}"}

    def parse_ollama_response(self, response_text: str, original_prompt: str) -> Dict:
        """Parsea la respuesta de Ollama para extraer elementos estructurados"""
        elements = {
            "subject": original_prompt,
            "action": "presente en la escena",
            "location": "entorno cinematográfico",
            "atmosphere": "atmósfera envolvente",
            "lighting_suggestion": "iluminación cinematográfica",
            "visual_effects": [],
            "audio_suggestions": []
        }
        # Buscar patrones en la respuesta
        lines = response_text.lower().split('\n')
        for line in lines:
            if 'sujeto' in line or 'protagonista' in line:
                elements["subject"] = line.split(':')[-1].strip()
            elif 'acción' in line or 'movimiento' in line:
                elements["action"] = line.split(':')[-1].strip()
            elif 'entorno' in line or 'lugar' in line or 'ubicación' in line:
                elements["location"] = line.split(':')[-1].strip()
            elif 'atmósfera' in line or 'ambiente' in line:
                elements["atmosphere"] = line.split(':')[-1].strip()
            elif 'iluminación' in line or 'luz' in line:
                elements["lighting_suggestion"] = line.split(':')[-1].strip()
            elif 'efecto' in line or 'visual' in line:
                effect = line.split(':')[-1].strip()
                if effect and effect not in elements["visual_effects"]:
                    elements["visual_effects"].append(effect)
            elif 'audio' in line or 'música' in line or 'sonido' in line:
                audio = line.split(':')[-1].strip()
                if audio and audio not in elements["audio_suggestions"]:
                    elements["audio_suggestions"].append(audio)
        return elements

    def analyze_prompt_with_ollama(self, prompt_json: str, model: str = "llama2") -> str:
        """Analiza un prompt JSON y sugiere mejoras usando Ollama"""
        if not self.ollama_available:
            return "Ollama no está disponible para análisis"
        if not model or model == "No disponible":
            return "Por favor, selecciona un modelo válido."
        
        # CORRECCIÓN: Usar solo el nombre base del modelo (sin tag)
        model_name = model.split(':')[0]
        
        try:
            analysis_prompt = f"""Analiza este prompt JSON para generación de video con Veo3 y sugiere mejoras:
{prompt_json}
Proporciona:
1. Fortalezas del prompt actual
2. Elementos que podrían mejorarse
3. Sugerencias específicas para hacer el video más impactante
4. Posibles elementos visuales o efectos adicionales
5. Recomendaciones de coherencia estilística
Sé específico y constructivo en tus sugerencias."""
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': analysis_prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error analizando prompt: {str(e)}"

    def generate_json_prompt(self, 
                           simple_prompt: str,
                           duration: int,
                           aspect_ratio: str,
                           shot_type: str,
                           camera_movement: str,
                           style: str,
                           lighting: str,
                           tone: str,
                           time_of_day: str,
                           add_audio: bool,
                           add_dialogue: bool,
                           dialogue_text: str,
                           prohibited_elements: str,
                           use_ai_enhancement: bool,
                           ollama_enhanced_elements: Optional[Dict] = None) -> str:
        """Genera el prompt JSON completo para Veo3"""
        # Si hay elementos mejorados por Ollama, usarlos
        if ollama_enhanced_elements and not ollama_enhanced_elements.get("error"):
            subject_description = ollama_enhanced_elements.get("subject", simple_prompt)
            action_description = ollama_enhanced_elements.get("action", "presente en la escena")
            location_description = ollama_enhanced_elements.get("location", "entorno cinematográfico")
            atmosphere = ollama_enhanced_elements.get("atmosphere", "atmósfera neutral")
            visual_effects = ollama_enhanced_elements.get("visual_effects", [])
        else:
            # Parsear el prompt simple normalmente
            elements = self.parse_simple_prompt(simple_prompt)
            subject_description = elements.get("subject", simple_prompt.split()[0] if simple_prompt else "sujeto")
            action_description = elements.get("action", "presente en la escena")
            location_description = elements.get("location", "entorno natural")
            atmosphere = f"atmósfera {elements.get('mood', 'neutral')}" if elements.get('mood') else "atmósfera neutral"
            visual_effects = []

        # Construir el JSON base
        json_prompt = {
            "model": "veo-3.0-fast",
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "shot": {
                "composition": self.shot_types.get(shot_type, "medium shot"),
                "camera_motion": self.camera_movements.get(camera_movement, "static"),
                "frame_rate": "24 fps"
            },
            "subject": {
                "primary": subject_description,
                "action": action_description,
                "physics": "realistic gravity and object interaction"
            },
            "scene": {
                "location": location_description,
                "time_of_day": self.times_of_day.get(time_of_day, "day"),
                "environment": atmosphere
            },
            "cinematography": {
                "lighting": self.lighting_types.get(lighting, "natural light"),
                "style": self.styles.get(style, "cinematic"),
                "tone": self.tones.get(tone, "neutral")
            },
            "visual_rules": {
                "prohibited_elements": prohibited_elements.split(",") if prohibited_elements else [],
                "auto_tone": False
            }
        }

        # Añadir efectos visuales si los hay
        if visual_effects:
            json_prompt["visual_details"] = {
                "effects": visual_effects[:5]  # Limitar a 5 efectos
            }

        # Añadir audio si está habilitado
        if add_audio:
            audio_suggestions = ollama_enhanced_elements.get("audio_suggestions", []) if ollama_enhanced_elements else []
            json_prompt["audio"] = {
                "ambient": audio_suggestions[:2] if audio_suggestions else ["sonido ambiente natural"],
                "soundtrack": "música suave instrumental",
                "sfx": ["efectos sutiles"],
                "mix_level": "balanced"
            }

        # Añadir diálogo si está habilitado
        if add_dialogue and dialogue_text:
            json_prompt["dialogue"] = {
                "script": dialogue_text,
                "voice": "voz clara y natural",
                "subtitles": False
            }

        # Mejorar con OpenAI si está habilitado
        if use_ai_enhancement and openai.api_key:
            json_prompt = self.enhance_with_ai(simple_prompt, json_prompt)

        return json.dumps(json_prompt, indent=2, ensure_ascii=False)

    def parse_simple_prompt(self, prompt: str) -> Dict:
        """Analiza un prompt simple y extrae elementos clave"""
        elements = {
            "subject": "",
            "action": "",
            "location": "",
            "mood": "",
            "details": []
        }
        # Análisis básico del prompt
        words = prompt.lower().split()
        # Palabras clave para diferentes elementos
        action_words = ["caminando", "corriendo", "volando", "saltando", "mirando", 
                       "sentado", "parado", "bailando", "luchando", "conduciendo",
                       "jugando", "trabajando", "leyendo", "escribiendo", "cantando"]
        location_words = ["ciudad", "bosque", "playa", "montaña", "desierto", 
                         "espacio", "habitación", "calle", "parque", "edificio",
                         "jardín", "oficina", "cocina", "laboratorio", "castillo"]
        mood_words = ["dramático", "feliz", "triste", "misterioso", "épico", 
                     "terrorífico", "romántico", "nostálgico", "tenso", "pacífico"]

        # Extraer elementos
        for i, word in enumerate(words):
            if word in action_words:
                elements["action"] = word
            if word in location_words:
                elements["location"] = word
            if word in mood_words:
                elements["mood"] = word

        # El sujeto generalmente está al principio
        if len(words) > 0:
            elements["subject"] = " ".join(words[:3])

        return elements

    def enhance_with_ai(self, simple_prompt: str, base_json: Dict) -> Dict:
        """Mejora el JSON usando OpenAI (opcional)"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en crear prompts JSON para Veo3. Mejora y detalla el siguiente JSON basándote en el prompt del usuario."},
                    {"role": "user", "content": f"Prompt original: {simple_prompt}\nJSON base: {json.dumps(base_json)}\nMejora este JSON añadiendo más detalles cinematográficos, efectos visuales apropiados y refinando las descripciones."}
                ]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print("❌ Error al mejorar con OpenAI:", e)
            return base_json


# Crear instancia del generador
generator = Veo3PromptGenerator()


# Funciones para Gradio
def generate_creative_ideas(base_prompt, ollama_model, num_ideas):
    """Genera ideas creativas usando Ollama"""
    if not generator.ollama_available:
        return "❌ Ollama no está instalado o no está ejecutándose.\nPara instalar:\n1. Visita https://ollama.ai/download\n2. Instala Ollama\n3. Ejecuta: ollama pull llama2"
    ideas = generator.generate_ideas_with_ollama(base_prompt, ollama_model, num_ideas)
    return "\n".join(ideas)


def enhance_with_ollama(simple_prompt, style, tone, ollama_model):
    """Mejora el prompt usando Ollama"""
    if not generator.ollama_available:
        return None, "❌ Ollama no disponible"
    enhanced = generator.enhance_prompt_with_ollama(simple_prompt, style, tone, ollama_model)
    if enhanced.get("error"):
        return None, f"❌ Error: {enhanced['error']}"
    # Crear un resumen de las mejoras
    summary = f"""✅ Prompt mejorado con Ollama ({ollama_model}):
**Sujeto mejorado:** {enhanced.get('subject', 'N/A')}
**Acción detallada:** {enhanced.get('action', 'N/A')}
**Ubicación:** {enhanced.get('location', 'N/A')}
**Atmósfera:** {enhanced.get('atmosphere', 'N/A')}
**Sugerencia de iluminación:** {enhanced.get('lighting_suggestion', 'N/A')}
**Efectos visuales:** {', '.join(enhanced.get('visual_effects', [])) or 'N/A'}
**Audio sugerido:** {', '.join(enhanced.get('audio_suggestions', [])) or 'N/A'}"""
    return enhanced, summary


def analyze_json_with_ollama(json_text, ollama_model):
    """Analiza el JSON generado con Ollama"""
    if not generator.ollama_available:
        return "❌ Ollama no disponible"
    return generator.analyze_prompt_with_ollama(json_text, ollama_model)


def generate_veo3_prompt(simple_prompt, duration, aspect_ratio, shot_type, 
                        camera_movement, style, lighting, tone, time_of_day,
                        add_audio, add_dialogue, dialogue_text, prohibited_elements,
                        use_ai_enhancement, ollama_enhanced_elements=None):
    """Función wrapper para Gradio"""
    return generator.generate_json_prompt(
        simple_prompt, duration, aspect_ratio, shot_type,
        camera_movement, style, lighting, tone, time_of_day,
        add_audio, add_dialogue, dialogue_text, prohibited_elements,
        use_ai_enhancement, ollama_enhanced_elements
    )


# Crear interfaz Gradio mejorada
def create_interface():
    with gr.Blocks(
        title="Generador de Prompts JSON para Veo3 + Ollama",
        theme=gr.themes.Monochrome(),
        css="""
            .gr-dropdown { color: black !important; }
            .gr-dropdown ul { background-color: white !important; color: black !important; }
            .gr-dropdown li:hover { background-color: #f0f0f0 !important; }
        """
    ) as demo:
        # Estado para almacenar elementos mejorados
        ollama_enhanced_state = gr.State(None)

        gr.Markdown("""
        # 🎬 Generador de Prompts JSON para Veo3 con IA Local
        Convierte tus ideas simples en prompts JSON estructurados para Veo3 de Google.
        Ahora con soporte de **Ollama** para IA local sin límites.
        ## 📝 Instrucciones:
        1. Escribe una descripción simple o usa Ollama para generar ideas
        2. Mejora tu prompt con IA local (opcional)
        3. Ajusta los parámetros cinematográficos
        4. Genera tu JSON optimizado
        """)

        # Verificación de Ollama
        with gr.Row():
            with gr.Column():
                # Manejo robusto de modelos
                model_choices = generator.ollama_models if generator.ollama_available and generator.ollama_models else ["llama2", "mistral", "codellama"]
                default_model = model_choices[0] if model_choices else "No disponible"

                ollama_model = gr.Dropdown(
                    label="Modelo de Ollama",
                    choices=model_choices,
                    value=default_model,
                    interactive=generator.ollama_available
                )

        with gr.Tab("🎯 Generador"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Sección Ollama
                    with gr.Accordion("🤖 Asistente IA Local (Ollama)", open=True):
                        # Generador de ideas
                        gr.Markdown("### 💡 Generador de Ideas")
                        idea_prompt = gr.Textbox(
                            label="Concepto base para ideas",
                            placeholder="Ej: transformación mágica, encuentro inesperado, viaje épico...",
                            lines=1
                        )
                        num_ideas = gr.Slider(
                            label="Número de ideas",
                            minimum=3,
                            maximum=10,
                            value=5,
                            step=1
                        )
                        generate_ideas_btn = gr.Button("💡 Generar Ideas Creativas", size="sm")
                        ideas_output = gr.Textbox(
                            label="Ideas Generadas",
                            lines=8,
                            interactive=False
                        )
                    # Input principal
                    gr.Markdown("### 📝 Tu Prompt")
                    simple_prompt = gr.Textbox(
                        label="Describe tu video",
                        placeholder="Ej: Un astronauta caminando en la luna con la Tierra de fondo",
                        lines=3
                    )
                    # Botón para mejorar con Ollama
                    enhance_ollama_btn = gr.Button("🚀 Mejorar con Ollama", variant="secondary", size="sm")
                    ollama_enhancement_output = gr.Markdown()
                    # Parámetros básicos
                    with gr.Row():
                        duration = gr.Slider(
                            label="⏱️ Duración (segundos)",
                            minimum=5,
                            maximum=60,
                            value=15,
                            step=1
                        )
                        aspect_ratio = gr.Radio(
                            label="📐 Formato",
                            choices=["16:9", "9:16", "1:1"],
                            value="16:9"
                        )
                    # Parámetros de cámara
                    gr.Markdown("### 🎥 Configuración de Cámara")
                    with gr.Row():
                        shot_type = gr.Dropdown(
                            label="Tipo de Shot",
                            choices=list(generator.shot_types.keys()),
                            value="Medium Shot"
                        )
                        camera_movement = gr.Dropdown(
                            label="Movimiento de Cámara",
                            choices=list(generator.camera_movements.keys()),
                            value="Estático"
                        )
                    # Estilo visual
                    gr.Markdown("### 🎨 Estilo Visual")
                    with gr.Row():
                        style = gr.Dropdown(
                            label="Estilo",
                            choices=list(generator.styles.keys()),
                            value="Cinematográfico"
                        )
                        lighting = gr.Dropdown(
                            label="Iluminación",
                            choices=list(generator.lighting_types.keys()),
                            value="Luz Natural Suave"
                        )
                    with gr.Row():
                        tone = gr.Dropdown(
                            label="Tono Emocional",
                            choices=list(generator.tones.keys()),
                            value="Misterioso"
                        )
                        time_of_day = gr.Dropdown(
                            label="Momento del Día",
                            choices=list(generator.times_of_day.keys()),
                            value="Atardecer"
                        )
                    # Opciones adicionales
                    gr.Markdown("### ⚙️ Opciones Adicionales")
                    add_audio = gr.Checkbox(label="🔊 Añadir Audio", value=True)
                    add_dialogue = gr.Checkbox(label="💬 Añadir Diálogo", value=False)
                    dialogue_text = gr.Textbox(
                        label="Texto del Diálogo",
                        placeholder="Escribe el diálogo aquí...",
                        visible=False
                    )
                    prohibited_elements = gr.Textbox(
                        label="🚫 Elementos Prohibidos (separados por comas)",
                        placeholder="texto, personas, elementos modernos"
                    )
                    use_ai_enhancement = gr.Checkbox(
                        label="🤖 Mejorar con OpenAI (requiere API key)",
                        value=False
                    )
                    # Botón de generación
                    generate_btn = gr.Button("🚀 Generar JSON", variant="primary", size="lg")
                with gr.Column(scale=1):
                    # Output
                    json_output = gr.Code(
                        label="📋 JSON Generado para Veo3",
                        language="json",
                        lines=25
                    )
                    # Análisis con Ollama
                    with gr.Accordion("🔍 Analizar JSON con Ollama", open=False):
                        analyze_btn = gr.Button("🔍 Analizar y Sugerir Mejoras", size="sm")
                        analysis_output = gr.Textbox(
                            label="Análisis y Sugerencias",
                            lines=10,
                            interactive=False
                        )
                    # Botones de acción
                    with gr.Row():
                        copy_btn = gr.Button("📋 Copiar JSON", size="sm")
                        download_btn = gr.Button("💾 Descargar JSON", size="sm")

        with gr.Tab("💡 Ideas Predefinidas"):
            gr.Markdown("""
            ### Prompts Creativos de Ejemplo
            Selecciona una idea y úsala como base para tu video.
            """)
            # Grid de ideas creativas
            with gr.Row():
                for i in range(0, 5):
                    with gr.Column():
                        gr.Button(
                            generator.creative_prompts[i],
                            size="sm",
                            elem_id=f"creative_{i}"
                        ).click(
                            lambda x=generator.creative_prompts[i]: x,
                            outputs=[simple_prompt]
                        )
            with gr.Row():
                for i in range(5, 10):
                    with gr.Column():
                        gr.Button(
                            generator.creative_prompts[i],
                            size="sm",
                            elem_id=f"creative_{i}"
                        ).click(
                            lambda x=generator.creative_prompts[i]: x,
                            outputs=[simple_prompt]
                        )

        with gr.Tab("📚 Ejemplos"):
            gr.Markdown("""
            ### Ejemplos de Prompts con JSON Completo
            Aquí encontrarás ejemplos de diferentes tipos de videos con sus prompts JSON correspondientes.
            """)
            # Ejemplo 1: Anuncio
            with gr.Accordion("🛍️ Ejemplo: Anuncio de Producto", open=False):
                gr.Code("""
{
  "model": "veo-3.0-fast",
  "duration": 8,
  "aspect_ratio": "16:9",
  "shot": {
    "composition": "close-up",
    "camera_motion": "slow dolly in",
    "frame_rate": "24 fps"
  },
  "subject": {
    "primary": "botella de perfume de cristal elegante",
    "action": "flores creciendo mágicamente alrededor",
    "physics": "realistic with magical elements"
  },
  "scene": {
    "location": "studio minimalista blanco",
    "time_of_day": "timeless",
    "environment": "atmósfera etérea con partículas doradas"
  },
  "cinematography": {
    "lighting": "soft studio lighting with rim light",
    "style": "editorial premium",
    "tone": "luxury and elegance"
  },
  "visual_details": {
    "props": ["pétalos flotantes", "partículas doradas"],
    "effects": ["slow motion", "depth of field", "lens flare sutil"]
  },
  "audio": {
    "ambient": ["suave brisa"],
    "soundtrack": "música ambient elegante",
    "sfx": ["whoosh suave", "tintineo cristalino"]
  },
  "visual_rules": {
    "prohibited_elements": ["texto", "personas", "logos"]
  }
}
                """, language="json")

            with gr.Accordion("🎬 Ejemplo: Escena de Acción", open=False):
                gr.Code("""
{
  "model": "veo-3.0-fast",
  "duration": 10,
  "aspect_ratio": "16:9",
  "shot": {
    "composition": "low angle tracking",
    "camera_motion": "fast paced following",
    "frame_rate": "60 fps"
  },
  "subject": {
    "primary": "coche deportivo negro mate",
    "action": "derrapando en curva cerrada bajo la lluvia",
    "physics": "realistic with dynamic motion"
  },
  "scene": {
    "location": "ciudad nocturna con neones",
    "time_of_day": "night",
    "environment": "lluvia intensa, reflejos en el asfalto",
    "weather": "heavy rain"
  },
  "cinematography": {
    "lighting": "high contrast neon lighting",
    "style": "cinematic action",
    "tone": "intense and thrilling"
  },
  "visual_details": {
    "effects": ["motion blur", "rain splashes", "sparks from wheels", "neon reflections"]
  },
  "audio": {
    "ambient": ["lluvia intensa", "truenos distantes"],
    "soundtrack": "música electrónica intensa",
    "sfx": ["motor rugiendo", "neumáticos chirriando", "salpicaduras"]
  }
}
                """, language="json")

        with gr.Tab("📖 Guía"):
            gr.Markdown("""
            ### 🎯 Guía de Uso con Ollama
            #### 🤖 Configurar Ollama:
            1. **Instalar Ollama:**
               ```bash
               # Linux/Mac
               curl -fsSL https://ollama.ai/install.sh | sh
               # Windows
               # Descarga desde https://ollama.ai/download
               ```
            2. **Descargar modelos:**
               ```bash
               # Modelos recomendados
               ollama pull llama2       # Modelo base equilibrado
               ollama pull mistral      # Rápido y eficiente
               ollama pull codellama    # Bueno para JSON
               ```
            3. **Verificar instalación:**
               ```bash
               ollama list
               ```
            #### 💡 Usar el Generador de Ideas:
            1. Escribe un concepto base (ej: "transformación")
            2. Selecciona cuántas ideas quieres
            3. Haz clic en "Generar Ideas Creativas"
            4. Usa las ideas como inspiración
            #### 🚀 Mejorar Prompts con Ollama:
            1. Escribe tu prompt básico
            2. Selecciona estilo y tono deseados
            3. Haz clic en "Mejorar con Ollama"
            4. El sistema añadirá detalles cinematográficos
            #### 🔍 Analizar JSON Generado:
            1. Genera tu JSON
            2. Abre el acordeón "Analizar JSON"
            3. Haz clic en "Analizar y Sugerir Mejoras"
            4. Recibe feedback detallado
            #### 💡 Tips para Mejores Resultados:
            - **Llama2**: Mejor para descripciones creativas
            - **Mistral**: Más rápido, bueno para iteraciones
            - **CodeLlama**: Excelente para estructurar JSON
            - Experimenta con diferentes modelos
            - Combina ideas generadas con tus propias visiones
            """)

        with gr.Tab("⚙️ Configuración"):
            gr.Markdown("""
            ### Configuración Avanzada
            #### 🤖 Modelos de Ollama Recomendados:
            | Modelo | Tamaño | Uso Recomendado | Comando |
            |--------|--------|-----------------|---------|
            | llama2 | 3.8GB | Propósito general | `ollama pull llama2` |
            | mistral | 4.1GB | Rápido y eficiente | `ollama pull mistral` |
            | codellama | 3.8GB | Estructurar JSON | `ollama pull codellama` |
            | neural-chat | 4.1GB | Conversacional | `ollama pull neural-chat` |
            #### 🎛️ Personalización:
            Puedes modificar las opciones predefinidas editando las listas en la clase `Veo3PromptGenerator`.
            #### 🔧 Solución de Problemas:
            - **Ollama no detectado**: Asegúrate de que el servicio esté ejecutándose
            - **Modelo no disponible**: Descárgalo con `ollama pull [nombre]`
            - **Respuestas lentas**: Prueba modelos más pequeños como Mistral
            """)

        # Event handlers
        def toggle_dialogue_input(add_dialogue):
            return gr.update(visible=add_dialogue)

        # Generar ideas con Ollama
        generate_ideas_btn.click(
            generate_creative_ideas,
            inputs=[idea_prompt, ollama_model, num_ideas],
            outputs=[ideas_output]
        )

        # Mejorar con Ollama
        enhance_ollama_btn.click(
            enhance_with_ollama,
            inputs=[simple_prompt, style, tone, ollama_model],
            outputs=[ollama_enhanced_state, ollama_enhancement_output]
        )

        # Analizar JSON con Ollama
        analyze_btn.click(
            analyze_json_with_ollama,
            inputs=[json_output, ollama_model],
            outputs=[analysis_output]
        )

        add_dialogue.change(
            toggle_dialogue_input,
            inputs=[add_dialogue],
            outputs=[dialogue_text]
        )

        # Generar JSON principal
        def generate_with_state(simple_prompt, duration, aspect_ratio, shot_type,
                              camera_movement, style, lighting, tone, time_of_day,
                              add_audio, add_dialogue, dialogue_text, prohibited_elements,
                              use_ai_enhancement, enhanced_state):
            return generate_veo3_prompt(
                simple_prompt, duration, aspect_ratio, shot_type,
                camera_movement, style, lighting, tone, time_of_day,
                add_audio, add_dialogue, dialogue_text, prohibited_elements,
                use_ai_enhancement, enhanced_state
            )

        generate_btn.click(
            generate_with_state,
            inputs=[
                simple_prompt, duration, aspect_ratio, shot_type,
                camera_movement, style, lighting, tone, time_of_day,
                add_audio, add_dialogue, dialogue_text, prohibited_elements,
                use_ai_enhancement, ollama_enhanced_state
            ],
            outputs=[json_output]
        )

        # Copiar al portapapeles
        copy_btn.click(
            None,
            inputs=[json_output],
            outputs=[],
            js="""
            (json_text) => {
                navigator.clipboard.writeText(json_text);
                alert('JSON copiado al portapapeles!');
            }
            """
        )

        # Auto-refresh de modelos cada 30 segundos
        def refresh_ollama_status():
            generator.ollama_available = generator.check_ollama()
            generator.ollama_models = generator.get_ollama_models() if generator.ollama_available else []
            return gr.update(
                choices=generator.ollama_models if generator.ollama_available else ["No disponible"],
                value=generator.ollama_models[0] if generator.ollama_models else "No disponible"
            )

    return demo


# Ejecutar la aplicación
if __name__ == "__main__":
    # Verificar e informar sobre Ollama
    if not generator.ollama_available:
        print("\n⚠️  Ollama no está instalado o no está ejecutándose.")
        print("Para instalarlo:")
        print("1. Visita https://ollama.ai/download")
        print("2. Instala Ollama para tu sistema")
        print("3. Ejecuta: ollama pull llama2")
        print("\nLa aplicación funcionará sin Ollama, pero con funciones limitadas.\n")
    else:
        print(f"\n✅ Ollama detectado con {len(generator.ollama_models)} modelos disponibles")
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="localhost",
        server_port=7860,
        debug=True
    )