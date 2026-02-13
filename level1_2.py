
import pygame
import cv2
import mediapipe as mp
import numpy as np
import joblib
import math
import random
import os
import sys

def resource_path(relative_path):
    """ Функция для работы путей при сборке в .exe """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path(os.path.join('models', 'kazakh_sign_model.pkl'))


WIDTH, HEIGHT = 1100, 720  
PANEL_HEIGHT = 150         
GAME_HEIGHT = HEIGHT - PANEL_HEIGHT
FPS = 60

# Цветовая палитра
C_BG = (15, 15, 20)
C_PANEL = (25, 25, 35)
C_ACCENT = (0, 255, 255)      # Голубой неон
C_GOLD = (255, 215, 0)        # Золото
C_SUCCESS = (50, 255, 100)    # Зеленый успех
C_ERROR = (255, 80, 80)       # Красный 
C_TEXT = (240, 240, 255)

LEVEL_LETTERS = ["А", "Ә", "Б", "В", "Г"]

class Particle:
    """ Эффекты частиц при успехе """
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-4, 0)
        self.life = 1.0
        self.size = random.randint(3, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1
        self.life -= 0.02

    def draw(self, surf):
        if self.life > 0:
            s = int(self.size * self.life)
            pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), s)

class SignQuestGame:
    def __init__(self, screen, clf, class_names):
        self.screen = screen
        self.clf = clf
        self.class_names = class_names 
        
        self.font_big = pygame.font.SysFont("Arial", 80, bold=True)
        self.font_mid = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)
        
        self.state = "TRIAL_1" 
        self.particles = []
        self.pulse = 0.0      
        
     
        self.stones = {l: False for l in LEVEL_LETTERS}
        self.current_target = None
        self.sequence_target = []   
        self.sequence_player = []   
        self.hold_timer = 0
        self.stage_progress = 0    
        
        self.assets = self.load_assets()
        self.btn_rect = pygame.Rect(WIDTH - 220, GAME_HEIGHT - 70, 200, 50)

    def load_assets(self):
        imgs = {}
        def load(name, w, h):
            path = resource_path(f"assets/{name}")
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, (w, h))
            return None 

        imgs['bg'] = load("bg_1.png", WIDTH, GAME_HEIGHT)
        imgs['guardian'] = load("guardian.png", 600, 450)
        return imgs

    def get_target_letter(self):
        """ Возвращает текущую букву, которую нужно показать """
        # Уровень 1
        if self.state == "TRIAL_1":
            for l in LEVEL_LETTERS:
                if not self.stones[l]: return l
        
        # Уровень 2 и 4
        elif self.state in ["TRIAL_2", "TRIAL_4"]:
            return self.current_target
            
        # Уровень 3 
        elif self.state == "TRIAL_3":
            if len(self.sequence_player) < len(self.sequence_target):
                return self.sequence_target[len(self.sequence_player)]
        return None

    def show_video_hint(self):
        """ Показ видео-подсказки """
        target = self.get_target_letter()
        if not target: return
        path = resource_path(f"assets/video_hints/{target}.mp4")
        if not os.path.exists(path): return

        cap = cv2.VideoCapture(path)
        playing = True
        while playing:
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            surf = pygame.surfarray.make_surface(frame)
            surf = pygame.transform.flip(surf, True, False)
            
            self.screen.fill((0,0,0))
            self.screen.blit(surf, (WIDTH//2 - 320, HEIGHT//2 - 240))
            msg = self.font_small.render("ЖАБУ ҮШІН КЕЗ КЕЛГЕН ПЕРНЕНІ БАСЫҢЫЗ", True, (200, 200, 200))
            self.screen.blit(msg, (WIDTH//2 - msg.get_width()//2, HEIGHT - 80))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type in [pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.QUIT]:
                    playing = False
        cap.release()

    def update_logic(self, pred, conf):
        self.pulse += 0.05
        target = self.get_target_letter()
        
       
        if self.state in ["TRIAL_2", "TRIAL_4"] and not self.current_target:
            self.current_target = random.choice(LEVEL_LETTERS)

        # Проверка совпадения
        is_match = (pred == target and conf > 0.5)

        # УРОВЕНЬ 1
        if self.state == "TRIAL_1":
            if is_match:
                self.hold_timer += 1
                if self.hold_timer > 20: 
                    self.stones[target] = True
                    self.spawn_particles(WIDTH//2, GAME_HEIGHT//2, C_GOLD)
                    self.hold_timer = 0
            else: self.hold_timer = 0
            
            if all(self.stones.values()): 
                self.next_state("TRIAL_2")

        #УРОВЕНЬ 2
        elif self.state == "TRIAL_2": 
            if is_match:
                self.hold_timer += 1
                if self.hold_timer > 30: 
                    self.stage_progress += 1
                    self.spawn_particles(WIDTH//2, GAME_HEIGHT//2, C_ACCENT)
                    self.current_target = random.choice(LEVEL_LETTERS) # Новая буква
                    self.hold_timer = 0
            else: 
                self.hold_timer = max(0, self.hold_timer - 0.5)
            
            if self.stage_progress >= 3: 
                self.next_state("TRIAL_3")

        # УРОВЕНЬ 3
        elif self.state == "TRIAL_3": 
            
            if not self.sequence_target: 
                self.sequence_target = random.sample(LEVEL_LETTERS, 3)
            
            if is_match and target:
                self.hold_timer += 1
                if self.hold_timer > 20:
                    self.sequence_player.append(target)
                    self.spawn_particles(WIDTH//2, GAME_HEIGHT//2, C_SUCCESS)
                    self.hold_timer = 0
            else: 
                self.hold_timer = 0
            
            
            if len(self.sequence_player) == len(self.sequence_target):
                self.next_state("TRIAL_4")

        # УРОВЕНЬ 4
        elif self.state == "TRIAL_4":
            if is_match:
                self.hold_timer += 1
                if self.hold_timer > 15: 
                    self.stage_progress += 1
                    self.spawn_particles(WIDTH//2, GAME_HEIGHT//2, C_ERROR) 
                    self.current_target = random.choice(LEVEL_LETTERS)
                    self.hold_timer = 0
            else:
                self.hold_timer = max(0, self.hold_timer - 0.5)

            if self.stage_progress >= 5:
                self.state = "WIN"

    def next_state(self, new_state):
        self.state = new_state
        self.stage_progress = 0
        self.hold_timer = 0
        self.sequence_player = []
        self.sequence_target = []
        
       
        if new_state in ["TRIAL_2", "TRIAL_4"]:
            self.current_target = random.choice(LEVEL_LETTERS)
        else:
            self.current_target = None
            
        self.spawn_particles(WIDTH//2, GAME_HEIGHT//2, C_GOLD)

    def spawn_particles(self, x, y, color):
        for _ in range(25): self.particles.append(Particle(x, y, color))

    def draw_ui(self, pred, conf, lm):
        # 1. Фон
        if self.assets['bg']: self.screen.blit(self.assets['bg'], (0,0))
        else: self.screen.fill(C_BG)

        center_x, center_y = WIDTH // 2, GAME_HEIGHT // 2

        # --- ОТРИСОВКА УРОВНЕЙ ---
        
        # УРОВЕНЬ 1
        if self.state == "TRIAL_1":
            gap = 160
            start_x = center_x - (gap * 2)
            for i, l in enumerate(LEVEL_LETTERS):
                pos_x = start_x + i * gap
                active = self.stones[l]
                current = (l == self.get_target_letter())
                scale = 1.1 if current else 1.0
                if current: scale += math.sin(self.pulse) * 0.05
                rect = pygame.Rect(0, 0, 100 * scale, 140 * scale)
                rect.center = (pos_x, center_y)
                col = C_GOLD if active else (C_ACCENT if current else (60, 60, 70))
                pygame.draw.rect(self.screen, (30,30,40), rect, border_radius=15)
                pygame.draw.rect(self.screen, col, rect, 3 if not active else 0, border_radius=15)
                txt = self.font_big.render(l, True, col)
                self.screen.blit(txt, (rect.centerx - txt.get_width()//2, rect.centery - txt.get_height()//2))

        # УРОВЕНЬ 2
        elif self.state == "TRIAL_2":
            if self.assets['guardian']: 
                self.screen.blit(self.assets['guardian'], (WIDTH - 550, GAME_HEIGHT - 450))
            
            lvl_txt = self.font_small.render("2 КЕЗЕҢ:САҚШЫ", True, C_ACCENT)
            self.screen.blit(lvl_txt, (50, 30))

            target = self.get_target_letter() or "?"
            t1 = self.font_mid.render("МЫНА ӘРІПТІ КӨРСЕТ:", True, C_TEXT)
            t2 = self.font_big.render(target, True, C_GOLD)
            
            # Анимация
            s = 1.0 + math.sin(self.pulse)*0.1
            t2 = pygame.transform.rotozoom(t2, 0, s)

            self.screen.blit(t1, (100, center_y - 60))
            self.screen.blit(t2, (150, center_y + 10))
            
            prog = self.font_small.render(f"Сәтті: {self.stage_progress}/3", True, C_TEXT)
            self.screen.blit(prog, (100, center_y + 120))

      
        elif self.state == "TRIAL_3":
            lvl_txt = self.font_small.render("3 КЕЗЕҢ: ТІЗБЕК ", True, C_SUCCESS)
            self.screen.blit(lvl_txt, (50, 30))
            
           
            start_x = center_x - 150
            for i in range(3):
                pos_x = start_x + i * 120
                box_col = (80, 80, 90)
                char_to_draw = "?"

                if i < len(self.sequence_player):
                    box_col = C_SUCCESS # угадали
                    char_to_draw = self.sequence_player[i]
                elif i == len(self.sequence_player):
                    box_col = C_GOLD    # цель
                    char_to_draw = self.sequence_target[i] if self.sequence_target else "?"
                
                # Рисуем
                pygame.draw.rect(self.screen, box_col, (pos_x, center_y - 50, 100, 100), border_radius=10)
                pygame.draw.rect(self.screen, (255,255,255), (pos_x, center_y - 50, 100, 100), 2, border_radius=10)
                
                txt = self.font_big.render(char_to_draw, True, C_BG)
                self.screen.blit(txt, (pos_x + 30, center_y - 35))
            
            hint = self.font_small.render("Келесі әріпті жинаңыз", True, C_TEXT)
            self.screen.blit(hint, (center_x - hint.get_width()//2, center_y + 100))

        # УРОВЕНЬ 4
        elif self.state == "TRIAL_4":
            lvl_txt = self.font_small.render("4 КЕЗЕҢ: ФИНАЛ ", True, C_ERROR)
            self.screen.blit(lvl_txt, (50, 30))
            
            target = self.get_target_letter() or "!"
            
            # Тряска текста
            shake_x = random.randint(-2, 2)
            shake_y = random.randint(-2, 2)
            
            t1 = self.font_mid.render("ТЕЗ КӨРСЕТ!", True, C_TEXT)
            t2 = self.font_big.render(target, True, C_ERROR)
            
            self.screen.blit(t1, (center_x - t1.get_width()//2 + shake_x, center_y - 80 + shake_y))
            self.screen.blit(t2, (center_x - t2.get_width()//2 + shake_x, center_y + shake_y))
            
            prog = self.font_small.render(f"Қалды: {5 - self.stage_progress}", True, C_TEXT)
            self.screen.blit(prog, (center_x - prog.get_width()//2, center_y + 100))

        # ПОБЕДА
        elif self.state == "WIN":
            self.screen.fill((20, 40, 20))
            txt = self.font_big.render("ЖЕҢІС!", True, C_GOLD)
            sub = self.font_mid.render("БАРЛЫҚ ДЕҢГЕЙЛЕР ӨТІЛДІ", True, C_SUCCESS)
            self.screen.blit(txt, (center_x - txt.get_width()//2, center_y - 50))
            self.screen.blit(sub, (center_x - sub.get_width()//2, center_y + 50))

        # 3. Нижняя панель
        pygame.draw.rect(self.screen, C_PANEL, (0, HEIGHT - PANEL_HEIGHT, WIDTH, PANEL_HEIGHT))
        pygame.draw.line(self.screen, C_ACCENT, (0, HEIGHT - PANEL_HEIGHT), (WIDTH, HEIGHT - PANEL_HEIGHT), 4)
        
        # Кнопка видео
        pygame.draw.rect(self.screen, (50, 50, 70), self.btn_rect, border_radius=10)
        btn_txt = self.font_small.render("ВИДЕО КӨМЕК", True, C_TEXT)
        self.screen.blit(btn_txt, (self.btn_rect.x + 40, self.btn_rect.y + 15))

      
        y_off = HEIGHT - PANEL_HEIGHT + 25
        if lm: 
            status = f"Танылды: {pred if pred else '...'}  ({int(conf*100)}%)"
            col = C_SUCCESS if conf > 0.5 else (150, 150, 150)
            self.screen.blit(self.font_mid.render(status, True, col), (350, y_off))
            
           
            pygame.draw.rect(self.screen, (50, 50, 60), (350, y_off + 60, 400, 15), border_radius=10)
            fill = min(1.0, self.hold_timer / 30.0) 
            pygame.draw.rect(self.screen, C_ACCENT, (350, y_off + 60, int(400 * fill), 15), border_radius=10)
            
            # Скелет руки
            start_skel = (50, y_off - 10)
            for hand_lms in lm:
                for conn in mp.solutions.hands.HAND_CONNECTIONS:
                    p1, p2 = hand_lms.landmark[conn[0]], hand_lms.landmark[conn[1]]
                    pygame.draw.line(self.screen, C_SUCCESS, 
                                     (start_skel[0] + p1.x*220, start_skel[1] + p1.y*130),
                                     (start_skel[0] + p2.x*220, start_skel[1] + p2.y*130), 2)
        else:
            self.screen.blit(self.font_mid.render("ҚОЛЫҢЫЗДЫ КӨРСЕТІҢІЗ", True, (80, 80, 80)), (350, y_off + 30))

        #Частицы успеха праздновать 
        for p in self.particles:
            p.update()
            p.draw(self.screen)
        self.particles = [p for p in self.particles if p.life > 0]

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Qazaq Sign Quest")
    
    try:
        data = joblib.load(MODEL_PATH)
        clf = data['rf_model']
        class_names = data['letter_names']
    except Exception as e:
        print(f"Ошибка модели: {e}")
        pygame.quit()
        sys.exit()

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    game = SignQuestGame(screen, clf, class_names)
    clock = pygame.time.Clock()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        
        # предсказание 126 точек
        pred, conf = None, 0.0
        if results.multi_hand_landmarks:
            lh, rh = np.zeros(63), np.zeros(63)
            for i, hand_info in enumerate(results.multi_handedness):
                label = hand_info.classification[0].label
                coords = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[i].landmark]).flatten()
                if label == 'Left': lh = coords
                else: rh = coords
            
            features = np.concatenate([lh, rh]).reshape(1, -1)
            probs = clf.predict_proba(features)[0]
            best_idx = np.argmax(probs)
            pred = class_names[best_idx]
            conf = probs[best_idx]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.btn_rect.collidepoint(event.pos):
                    game.show_video_hint()

        game.update_logic(pred, conf)
        game.draw_ui(pred, conf, results.multi_hand_landmarks)
        
        pygame.display.flip()
        clock.tick(FPS)