# ============================================================
#  Child Safety Monitoring System v2.0 — Configuration
# ============================================================

CAMERA_LOCATION = "Main Entrance"

# --- Camera ---
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
TARGET_FPS    = 30

# --- Detection ---
YOLO_MODEL         = 'yolov8n.pt'
CONFIDENCE_THRESH  = 0.50
PERSON_CLASS_ID    = 0

# --- Alert thresholds ---
ALERT_FRAME_THRESHOLD  = 15       # was 8 — reduces false zone alerts
ALERT_COOLDOWN_SECONDS = 10      # seconds between repeated sound alerts

# --- Lying down detection ---
LYING_DOWN_SECONDS     = 30      # seconds person must be lying before 1122 alert
LYING_COOLDOWN_SECONDS = 60      # seconds between repeated lying alerts

# --- Age estimation ---
AGE_CHILD_THRESHOLD    = 15      # persons estimated under this age = child
AGE_ESTIMATION_INTERVAL = 15     # estimate age every N frames (saves CPU)

# --- Sound ---
BEEP_FREQUENCY = 1000
BEEP_DURATION  = 700

# ============================================================
#  EMAIL CONFIGURATION
#  Use Gmail App Passwords:
#  Google Account → Security → 2-Step → App Passwords
# ============================================================
EMAIL_ENABLED      = True
SENDER_EMAIL       = "your_sender_email@gmail.com"
SENDER_PASSWORD    = "your_app_password_here"
SMTP_SERVER        = "smtp.gmail.com"
SMTP_PORT          = 587
EMAIL_COOLDOWN_SECS = 30

# --- Alert recipients ---
FAMILY_EMAIL   = "family@gmail.com"          # Child's family
POLICE_EMAIL   = "police_station@gmail.com"  # Local police
RESCUE_EMAIL   = "rescue1122@gmail.com"      # 1122 rescue services

# ============================================================
#  WHATSAPP / SMS via TWILIO
#  Sign up free at: https://www.twilio.com
#  Get Account SID, Auth Token, and a Twilio phone number
# ============================================================
TWILIO_ENABLED       = False   # Set True after adding credentials
TWILIO_ACCOUNT_SID   = "your_account_sid_here"
TWILIO_AUTH_TOKEN    = "your_auth_token_here"
TWILIO_FROM_NUMBER   = "+1234567890"   # Your Twilio number

# WhatsApp recipients (prefix with whatsapp:)
FAMILY_WHATSAPP   = "whatsapp:+92300xxxxxxx"
POLICE_WHATSAPP   = "whatsapp:+92300xxxxxxx"
RESCUE_WHATSAPP   = "whatsapp:+92300xxxxxxx"

# SMS recipients (plain phone numbers)
FAMILY_PHONE   = "+92300xxxxxxx"
POLICE_PHONE   = "+92300xxxxxxx"
RESCUE_PHONE   = "+92300xxxxxxx"

# --- Snapshot ---
SNAPSHOT_DIR          = "incidents"    # folder to save incident photos
SNAPSHOT_QUALITY      = 92            # JPEG quality (0-100)
ATTACH_SNAPSHOT_EMAIL = True          # attach photo to alert emails

# --- Visual ---
ZONE_COLOR_SAFE     = (0, 200, 0)
ZONE_COLOR_DANGER   = (0, 0, 255)
ZONE_FILL_ALPHA     = 0.25
BBOX_COLOR_NORMAL   = (255, 180, 0)
BBOX_COLOR_ALERT    = (0, 0, 255)
BBOX_COLOR_CHILD    = (0, 165, 255)    # Orange for detected children
BBOX_COLOR_LYING    = (0, 0, 180)      # Dark red for lying person
ALERT_BANNER_COLOR  = (0, 0, 200)
FONT_SCALE          = 0.65
FONT_THICKNESS      = 2

# --- Weapon Model ---
WEAPON_MODEL = 'yolov8n-weapons.pt'   # separate model for guns + knives
WEAPON_CONFIDENCE_THRESH = 0.20        # good balance for knife detection

# --- Paths ---
LOG_FILE = "safety_log.txt"