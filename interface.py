import customtkinter as ctk
import tkinter as tk
from PIL import Image
import os
import threading  # Para processamento em paralelo
import queue      # Para comunicação thread-safe

# --- NOSSAS IMPORTAÇÕES DOS MODELOS ---
#from predict_bert import predict_texts as predict_relevance_bert
from predicao_aplicavel_assunto import predict_applicability

# --- Constantes de Cor ---
NEUTRAL_COLOR = "#E0E0E0"
LOW_COLOR = "#4CAF50"      # Verde (Baixa Relevância)
MEDIUM_COLOR = "#FFD700"   # Amarelo (Média Relevância)
HIGH_COLOR = "#FF6B6B"     # Vermelho (Alta Relevância)

APPLICABLE_COLOR = "#4CAF50"      # Verde (Aplicável)
NOT_APPLICABLE_COLOR = "#F44336"  # Vermelho (Não Aplicável)
WAITING_COLOR = "#B0B0B0"          # Cinza (Aguardando)

# Configuração inicial do CustomTkinter
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class TextClassifierApp(ctk.CTk):
    """
    Interface Gráfica para Classificação de Normativos (Relevância e Aplicabilidade).
    """
    def __init__(self):
        super().__init__()

        # --- Configurações da Janela ---
        self.title("NormaAI - Classificação de Normativos")
        self.geometry("1000x700")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Carregar Logo e Fila de Comunicação ---
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bndes_logo.png")
        self.logo_pil_image = Image.open(logo_path)
        self.result_queue = queue.Queue() # Fila Thread-Safe

        # --- Frame Principal (O "Cartão" Central) ---
        self.main_container = ctk.CTkFrame(self, corner_radius=15, fg_color="white")
        self.main_container.grid(row=0, column=0, padx=50, pady=50, sticky="nsew")
        self.main_container.grid_columnconfigure((0, 1), weight=1)
        self.main_container.grid_rowconfigure(2, weight=1)

        # --- Criação dos Componentes da UI ---
        self._create_header()
        self._create_input_widgets()
        self._create_result_widgets()

        # --- Estado Inicial e Listener da Fila ---
        self.set_neutral_state()
        self.check_result_queue() # Inicia o "ouvinte" da fila

    def _create_header(self):
        """Cria o cabeçalho (Título e Logo)."""
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, padx=30, pady=(30, 10), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Lógica para carregar a imagem da logo (se existir) ou usar texto como fallback
        if self.logo_pil_image:
            self.logo_ctk_image = ctk.CTkImage(light_image=self.logo_pil_image, dark_image=self.logo_pil_image, size=(160, 30))
            logo_label = ctk.CTkLabel(header_frame, image=self.logo_ctk_image, text="")
        else:
            logo_label = ctk.CTkLabel(header_frame, text="BNDES", font=ctk.CTkFont(size=20, weight="bold"), text_color="#006497")

        logo_label.grid(row=0, column=1, padx=(0, 0), sticky="e")

        title_label = ctk.CTkLabel(
            self.main_container,
            text="NormaAI - Classificação de Normativos",
            font=ctk.CTkFont(size=26, weight="normal"),
            anchor="w"
        )
        title_label.grid(row=0, column=0, columnspan=2, padx=30, pady=(50, 20), sticky="w")


    def _create_input_widgets(self):
        """Cria os widgets do lado esquerdo (Input de Texto e Botão)."""
        self.input_frame = ctk.CTkFrame(self.main_container, corner_radius=10, fg_color="transparent")
        # rowspan=2 para cobrir a área de resultados e alinhar verticalmente
        self.input_frame.grid(row=1, column=0, rowspan=2, padx=(30, 15), pady=(0, 30), sticky="nsew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure(2, weight=1) # Textbox expansível

        # Título
        input_title = ctk.CTkLabel(self.input_frame, text="1. Insira o Texto do Normativo", font=ctk.CTkFont(size=18, weight="bold"), anchor="w")
        input_title.grid(row=0, column=0, padx=(20, 0), pady=(10, 15), sticky="w")

        # Descrição
        description_label = ctk.CTkLabel(self.input_frame, text="Cole aqui o ASSUNTO do normativo para análise...", text_color="gray", anchor="w")
        description_label.grid(row=1, column=0, padx=(20, 0), pady=(0, 5), sticky="w")

        # Textbox para input
        self.text_input = ctk.CTkTextbox(self.input_frame, wrap="word", font=ctk.CTkFont(size=14), height=300, border_width=1, border_color="#C0C0C0")
        self.text_input.grid(row=2, column=0, padx=(20, 20), sticky="nsew", pady=(0, 5))

        # Label "Processando..." (invisível inicialmente)
        self.processing_label = ctk.CTkLabel(self.input_frame, text="Processando modelos (NN e BERT)...", text_color="#006497", font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
        self.processing_label.grid(row=3, column=0, padx=(20, 0), pady=(5, 10), sticky="w")
        self.processing_label.grid_remove() 
        
        # Botão de Classificação
        self.action_button = ctk.CTkButton(self.input_frame, text="Classificar Normativo", command=self.submit_classification_request, font=ctk.CTkFont(size=16, weight="bold"), text_color="#FFFFFF", height=45, corner_radius=8)
        self.action_button.grid(row=5, column=0, padx=(20, 20), pady=(5, 10), sticky="ew")

    def _create_result_widgets(self):
        """Cria os widgets do lado direito (Resultado)."""
        
        # Frame do Título
        self.text_frame = ctk.CTkFrame(self.main_container, corner_radius=10, fg_color="transparent")
        self.text_frame.grid(row=1, column=1, padx=(15, 30), pady=(0, 0), sticky="ew")
        result_title = ctk.CTkLabel(self.text_frame, text="2. Resultado da Classificação", font=ctk.CTkFont(size=18, weight="bold"), anchor="w")
        result_title.grid(row=0, column=0, padx=20, pady=(10, 15), sticky="w") 

        # Caixa de Resultados
        self.result_frame = ctk.CTkFrame(self.main_container, corner_radius=10, fg_color="#F8F8F8", border_width=1, border_color="#E0E0E0")
        self.result_frame.grid(row=2, column=1, padx=(15, 30), pady=(15, 30), sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)

        # Badge "APLICÁVEL"
        self.applicable_badge = ctk.CTkFrame(self.result_frame, corner_radius=50)
        self.applicable_badge.grid(row=1, column=0, padx=20, pady=(20, 20), sticky="")
        self.badge_label = ctk.CTkLabel(self.applicable_badge, text_color="white", font=ctk.CTkFont(size=14, weight="bold"), padx=10, pady=5)
        self.badge_label.grid(row=0, column=0, sticky="nsew")

        # Subtítulo Relevância
        relevance_title = ctk.CTkLabel(self.result_frame, text="Relevância:", font=ctk.CTkFont(size=16, weight="bold"), text_color="#505050")
        relevance_title.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="w")

        # Indicador de Relevância (Bolas)
        BALL_SIZE = 20
        self.relevance_balls_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        self.relevance_balls_frame.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="") # Centraliza

        self.ball_low = self._create_ball(self.relevance_balls_frame, BALL_SIZE)
        self.ball_low.grid(row=0, column=0, padx=(0, 4))
        self.ball_medium = self._create_ball(self.relevance_balls_frame, BALL_SIZE)
        self.ball_medium.grid(row=0, column=1, padx=4)
        self.ball_high = self._create_ball(self.relevance_balls_frame, BALL_SIZE)
        self.ball_high.grid(row=0, column=2, padx=(4, 0))

        # Texto da Relevância (BAIXA, MÉDIA, ALTA)
        self.relevance_text_label = ctk.CTkLabel(self.result_frame, font=ctk.CTkFont(size=18, weight="bold"))
        self.relevance_text_label.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="") # Centraliza

    def _create_ball(self, parent, size):
        """Cria um frame redondo (indicador visual)."""
        return ctk.CTkFrame(parent, width=size, height=size, corner_radius=size//2)

    def set_neutral_state(self):
        """Define os resultados para o estado inicial (cinza)."""
        # Relevância
        self.ball_low.configure(fg_color=NEUTRAL_COLOR)
        self.ball_medium.configure(fg_color=NEUTRAL_COLOR)
        self.ball_high.configure(fg_color=NEUTRAL_COLOR)
        self.relevance_text_label.configure(text="--", text_color="gray")
        
        # Aplicabilidade
        self.applicable_badge.configure(fg_color=WAITING_COLOR)
        self.badge_label.configure(text="Aguardando Classificação")
        
        # Botão e Status
        self.action_button.configure(text="Classificar Normativo", state="normal")
        self.processing_label.grid_remove()

    def submit_classification_request(self):
        """Pega o texto de input e inicia a thread de classificação."""
        text_content = self.text_input.get("1.0", "end-1c").strip()
        
        if not text_content:
            print("Nenhum texto inserido.")
            return

        print(f"Texto enviado para classificação: {text_content[:50]}...")
        
        # 1. Coloca a UI em estado de "carregando"
        self.action_button.configure(text="Classificando...", state="disabled")
        self.processing_label.grid()

        # 2. Inicia a tarefa dos modelos em uma thread separada
        threading.Thread(
            target=self.run_classification_model, 
            args=(text_content, self.result_queue),
            daemon=True
        ).start()

    def run_classification_model(self, text, queue):
        """(Roda em THREAD SEPARADA) Chama os modelos de ML/IA."""
        
        # --- PREDIÇÃO REAL ---
        
        # Predição de Relevância (BERT - Simulado/Removido)
        # relevance = predict_relevance_bert(text)[0] # Se ativado
        relevance = "baixa"

        # Predição de Aplicabilidade (NN)
        print("Thread: Iniciando predição do modelo NN (Aplicabilidade)...")
        try:
            is_applicable = predict_applicability(text) # Retorna True/False
        except Exception as e:
            print(f"Thread: ERRO ao rodar modelo NN: {e}")
            is_applicable = False 
        
        print(f"Thread: Predição concluída. Relevância={relevance}, Aplicável={is_applicable}")
        
        # 3. Coloca o resultado na fila para a thread principal
        result_data = {"relevance": relevance, "applicable": is_applicable}
        queue.put(result_data)

    def check_result_queue(self):
        """(Roda na THREAD PRINCIPAL) Verifica a fila de resultados periodicamente."""
        try:
            # Tenta pegar um resultado da fila (sem bloquear)
            result = self.result_queue.get(block=False)
            
            # 4. Atualiza a interface com o resultado
            print("Main: Resultado recebido da fila. Atualizando UI.")
            self.update_results_ui(result)
            
        except queue.Empty:
            pass # Nenhuma ação se a fila estiver vazia
        finally:
            # Re-agenda a verificação para daqui a 100ms
            self.after(100, self.check_result_queue)

    def update_results_ui(self, result):
        """(Roda na THREAD PRINCIPAL) Atualiza os widgets com os resultados."""
        relevance = result.get("relevance", "baixa")
        is_applicable = result.get("applicable", False)

        # --- Atualiza a Relevância (Bolas e Texto) ---
        self.ball_low.configure(fg_color=NEUTRAL_COLOR)
        self.ball_medium.configure(fg_color=NEUTRAL_COLOR)
        self.ball_high.configure(fg_color=NEUTRAL_COLOR)

        relevance_lower = str(relevance).lower()

        if relevance_lower == "baixa":
            self.ball_low.configure(fg_color=LOW_COLOR)
            self.relevance_text_label.configure(text="BAIXA", text_color=LOW_COLOR)
        
        elif relevance_lower == "média":
            self.ball_low.configure(fg_color=MEDIUM_COLOR)
            self.ball_medium.configure(fg_color=MEDIUM_COLOR)
            self.relevance_text_label.configure(text="MÉDIA", text_color=MEDIUM_COLOR)
            
        elif relevance_lower == "alta":
            self.ball_low.configure(fg_color=HIGH_COLOR)
            self.ball_medium.configure(fg_color=HIGH_COLOR)
            self.ball_high.configure(fg_color=HIGH_COLOR)
            self.relevance_text_label.configure(text="ALTA", text_color=HIGH_COLOR)

        # --- Atualiza a Aplicabilidade (Badge) ---
        if is_applicable:
            self.applicable_badge.configure(fg_color=APPLICABLE_COLOR)
            self.badge_label.configure(text="✓ APLICÁVEL AO BNDES")
        else:
            self.applicable_badge.configure(fg_color=NOT_APPLICABLE_COLOR)
            self.badge_label.configure(text="✕ NÃO APLICÁVEL")

        # 5. Reseta o estado do botão
        self.action_button.configure(text="Classificar Normativo", state="normal")
        self.processing_label.grid_remove()

# --- Execução Principal ---
if __name__ == "__main__":
    print("Iniciando carregamento dos modelos...")
    print("Carregamento concluído. Iniciando interface.")
    
    app = TextClassifierApp()
    app.mainloop()