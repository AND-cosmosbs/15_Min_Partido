# backend/supabase_client.py

import os
from supabase import create_client, Client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Faltan SUPABASE_URL o SUPABASE_KEY en las variables de entorno. "
        "Configúralas en Render (Environment Variables)."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
