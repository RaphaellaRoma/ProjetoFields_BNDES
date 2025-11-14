import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import os
import threading  # Para executar a rede neural em paralelo
import queue      # Para comunicar o resultado de volta para a interface
import time       # Usado APENAS para simular o tempo da rede neural
import random     # Usado APENAS para simular a resposta do modelo

# --- Constantes de Cor para o Design ---
NEUTRAL_COLOR = "#E0E0E0"
LOW_COLOR = "#FF6B6B"      # Vermelho para Baixa Relevância
MEDIUM_COLOR = "#FFD700"   # Amarelo para Média Relevância
HIGH_COLOR = "#4CAF50"     # Verde para Alta Relevância

APPLICABLE_COLOR = "#4CAF50"       # Verde
NOT_APPLICABLE_COLOR = "#F44336"  # Vermelho
WAITING_COLOR = "#B0B0B0"          # Cinza para "Aguardando"

# Configuração inicial
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class TextClassifierApp(ctk.CTk):
    """
    Interface moderna de Classificação de Normativos.
    """
    def __init__(self):
        super().__init__()

        # --- Configurações da Janela ---
        self.title("NormaAI - Classificação de Normativos")
        self.geometry("1000x700")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Fila de Comunicação (Thread-Safe) ---
        # A rede neural (thread) colocará o resultado aqui
        # A interface (main thread) lerá o resultado daqui
        self.result_queue = queue.Queue()

        # --- Frame Principal (O "Cartão" Central) ---
        self.main_container = ctk.CTkFrame(self, corner_radius=15, fg_color="white")
        self.main_container.grid(row=0, column=0, padx=50, pady=50, sticky="nsew")
        self.main_container.grid_columnconfigure((0, 1), weight=1)
        self.main_container.grid_rowconfigure(2, weight=1) 

        # --- Cabeçalho ---
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, padx=30, pady=(30, 10), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        logo_label = ctk.CTkLabel(
            header_frame, 
            text="● BNDES", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#006497"
        )
        logo_label.grid(row=0, column=0, padx=(0, 20), sticky="w")

        title_label = ctk.CTkLabel(
            self.main_container,
            text="NormaAI - Classificação de Normativos",
            font=ctk.CTkFont(size=24, weight="normal"),
            anchor="w"
        )
        title_label.grid(row=1, column=0, columnspan=2, padx=30, pady=(5, 20), sticky="w")

        # --- LEFT SIDE: 1. Insira o Texto ---
        self._create_input_widgets()
        
        # --- RIGHT SIDE: 2. Resultado da Classificação ---
        self._create_result_widgets()

        # --- Estado Inicial ---
        # Define o estado neutro (cinza) no início
        self.set_neutral_state()
        
        # Inicia o "ouvinte" da fila de resultados
        self.check_result_queue()

    def _create_input_widgets(self):
        """Cria os widgets do lado esquerdo (Input)."""
        self.input_frame = ctk.CTkFrame(self.main_container, corner_radius=10, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, padx=(30, 15), pady=(10, 30), sticky="nsew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure(2, weight=1)

        input_title = ctk.CTkLabel(
            self.input_frame,
            text="1. Insira o Texto do Normativo",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        )
        input_title.grid(row=0, column=0, pady=(0, 15), sticky="w")

        description_label = ctk.CTkLabel(
            self.input_frame,
            text="Cole aqui o ASSUNTO do normativo para análise...",
            text_color="gray",
            anchor="w"
        )
        description_label.grid(row=1, column=0, pady=(0, 5), sticky="w")

        self.text_input = ctk.CTkTextbox(
            self.input_frame,
            wrap="word",
            font=ctk.CTkFont(size=14),
            height=300,
            border_width=1,
            border_color="#C0C0C0"
        )
        self.text_input.grid(row=2, column=0, sticky="nsew", pady=(0, 5))

        # Label "Processando..." (será controlado por grid_remove/grid)
        self.processing_label = ctk.CTkLabel(
            self.input_frame,
            text="Processando na Rede Neural...",
            text_color="#006497", # Azul BNDES
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        # Começa escondido
        self.processing_label.grid(row=3, column=0, pady=(5, 10), sticky="w")
        self.processing_label.grid_remove() 
        
        
        self.action_button = ctk.CTkButton(
            self.input_frame,
            text="Classificar Normativo",
            command=self.submit_classification_request,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            corner_radius=8
        )
        self.action_button.grid(row=5, column=0, pady=(0, 10), sticky="ew")

    def _create_result_widgets(self):
        """Cria os widgets do lado direito (Resultado)."""
        self.result_frame = ctk.CTkFrame(self.main_container, corner_radius=10, fg_color="#F8F8F8", border_width=1, border_color="#E0E0E0")
        self.result_frame.grid(row=2, column=1, padx=(15, 30), pady=(10, 30), sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)

        result_title = ctk.CTkLabel(
            self.result_frame,
            text="2. Resultado da Classificação",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        )
        result_title.grid(row=0, column=0, padx=20, pady=(20, 20), sticky="w")

        # --- Badge "APLICÁVEL" ---
        self.applicable_badge = ctk.CTkFrame(self.result_frame, corner_radius=20)
        self.applicable_badge.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")
        
        self.badge_label = ctk.CTkLabel(
            self.applicable_badge,
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold"),
            padx=10,
            pady=5
        )
        self.badge_label.grid(row=0, column=0, sticky="nsew")

        # --- Subtítulo Relevância ---
        relevance_title = ctk.CTkLabel(
            self.result_frame,
            text="Relevância:",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#505050"
        )
        relevance_title.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="w")

        # --- Indicador de Relevância (Bolas) ---
        BALL_SIZE = 20
        self.relevance_balls_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        self.relevance_balls_frame.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="w")

        # Criamos 3 "bolas" (Frames redondos)
        self.ball_low = self._create_ball(self.relevance_balls_frame, BALL_SIZE)
        self.ball_low.grid(row=0, column=0, padx=(0, 4))
        
        self.ball_medium = self._create_ball(self.relevance_balls_frame, BALL_SIZE)
        self.ball_medium.grid(row=0, column=1, padx=4)

        self.ball_high = self._create_ball(self.relevance_balls_frame, BALL_SIZE)
        self.ball_high.grid(row=0, column=2, padx=(4, 0))

        # --- Texto da Relevância (BAIXA, MÉDIA, ALTA) ---
        self.relevance_text_label = ctk.CTkLabel(
            self.result_frame,
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.relevance_text_label.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="w")

    def _create_ball(self, parent, size):
        """Função helper para criar um frame redondo (bola)."""
        return ctk.CTkFrame(parent, width=size, height=size, corner_radius=size//2)

    def set_neutral_state(self):
        """Define os resultados para o estado inicial (neutro/cinza)."""
        
        # Reseta as bolas para a cor neutra
        self.ball_low.configure(fg_color=NEUTRAL_COLOR)
        self.ball_medium.configure(fg_color=NEUTRAL_COLOR)
        self.ball_high.configure(fg_color=NEUTRAL_COLOR)
        
        # Reseta o texto de relevância
        self.relevance_text_label.configure(text="--", text_color="gray")
        
        # Reseta o badge de aplicabilidade
        self.applicable_badge.configure(fg_color=WAITING_COLOR)
        self.badge_label.configure(text="Aguardando Classificação")
        
        # Garante que o botão esteja clicável
        self.action_button.configure(text="Classificar Normativo", state="normal")
        
        # Esconde a label "Processando..."
        self.processing_label.grid_remove()

    def submit_classification_request(self):
        """
        Função chamada pelo botão.
        Pega o texto e inicia a thread de classificação.
        """
        text_content = self.text_input.get("1.0", "end-1c").strip()
        
        if not text_content:
            print("Nenhum texto inserido.")
            return

        print(f"Texto enviado para classificação: {text_content[:50]}...")
        
        # 1. Coloca a UI em estado de "carregando"
        self.action_button.configure(text="Classificando...", state="disabled")
        self.processing_label.grid() # Mostra "Processando..."

        # 2. Inicia a tarefa da rede neural em uma thread separada
        #    Isso evita que a interface congele.
        threading.Thread(
            target=self.run_classification_model, 
            args=(text_content, self.result_queue),
            daemon=True # Garante que a thread feche se a janela principal fechar
        ).start()

    def run_classification_model(self, text, queue):
        """
        (Esta função roda em uma THREAD SEPARADA)
        Aqui é onde você chama seu outro módulo com a rede neural.
        """
        
        # --- SUBSTITUA ESTA SEÇÃO PELO SEU CÓDIGO REAL ---
        print("Thread: Iniciando simulação do modelo (BERT)...")
        # import seu_modulo_de_ia
        
        # Simula o tempo de processamento da rede neural (ex: 2 segundos)
        time.sleep(2) 
        
        # Simula um resultado aleatório
        relevance_levels = ["baixa", "média", "alta"]
        relevance = random.choice(relevance_levels)
        is_applicable = random.choice([True, False])
        
        # resultado_real = seu_modulo_de_ia.prever(text)
        # relevance = resultado_real['relevancia']
        # is_applicable = resultado_real['aplicabilidade']
        
        print(f"Thread: Simulação concluída. Relevância={relevance}, Aplicável={is_applicable}")
        # --- FIM DA SEÇÃO DE SUBSTITUIÇÃO ---
        
        
        # 3. Coloca o dicionário de resultados na fila
        result_data = {"relevance": relevance, "applicable": is_applicable}
        queue.put(result_data)

    def check_result_queue(self):
        """
        (Esta função roda na THREAD PRINCIPAL)
        Verifica a fila de resultados periodicamente (a cada 100ms).
        """
        try:
            # Tenta pegar um resultado da fila (sem bloquear)
            result = self.result_queue.get(block=False)
            
            # 4. Se um resultado for encontrado, atualiza a interface
            print("Main: Resultado recebido da fila. Atualizando UI.")
            self.update_results_ui(result)
            
        except queue.Empty:
            # Se a fila estiver vazia, não faz nada
            pass
        finally:
            # Re-agenda a verificação para daqui a 100ms
            self.after(100, self.check_result_queue)

    def update_results_ui(self, result):
        """
        (Esta função roda na THREAD PRINCIPAL)
        Atualiza os widgets da interface com os resultados recebidos.
        """
        relevance = result.get("relevance", "baixa")
        is_applicable = result.get("applicable", False)

        # --- Atualiza a Relevância (Bolas e Texto) ---
        
        # Primeiro, reseta todas para o neutro
        self.ball_low.configure(fg_color=NEUTRAL_COLOR)
        self.ball_medium.configure(fg_color=NEUTRAL_COLOR)
        self.ball_high.configure(fg_color=NEUTRAL_COLOR)

        if relevance == "baixa":
            self.ball_low.configure(fg_color=LOW_COLOR) # Acende 1 bola vermelha
            self.relevance_text_label.configure(text="BAIXA", text_color=LOW_COLOR)
        
        elif relevance == "média":
            self.ball_low.configure(fg_color=MEDIUM_COLOR) # Acende 2 bolas amarelas
            self.ball_medium.configure(fg_color=MEDIUM_COLOR)
            self.relevance_text_label.configure(text="MÉDIA", text_color=MEDIUM_COLOR)
            
        elif relevance == "alta":
            self.ball_low.configure(fg_color=HIGH_COLOR) # Acende 3 bolas verdes
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

        # 5. Reseta o estado do botão e esconde "Processando"
        self.action_button.configure(text="Classificar Normativo", state="normal")
        self.processing_label.grid_remove()

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    app = TextClassifierApp()
    app.mainloop()