<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# continue with phase 2

Here's the full Phase 2 implementation guide.

***

## Phase 2: Font Database Setup (Supabase)

### Step 1 — Get Your Google Fonts API Key

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project → name it `font-identifier`
3. In the left sidebar → **APIs \& Services → Library**
4. Search for **"Web Fonts Developer API"** → click **Enable**[^1]
5. Go to **APIs \& Services → Credentials → Create Credentials → API Key**
6. Copy the key → add it to `/web/.env.local` and `/model/.env`:

```env
GOOGLE_FONTS_API_KEY=your-google-fonts-api-key
```


***

### Step 2 — Enable pgvector in Supabase

1. Open your Supabase project → go to **SQL Editor**
2. Run this to enable the vector extension:[^2]

```sql
create extension if not exists vector;
```

3. Click **Run** — you should see "Success. No rows returned"

***

### Step 3 — Create Database Tables

Still in the SQL Editor, run each block below one at a time.

**`fonts` table — stores all font metadata + embeddings:**

```sql
create table fonts (
  id        serial primary key,
  name      text not null,
  family    text,
  category  text,        -- serif, sans-serif, monospace, display, handwriting
  variants  text[],      -- ['regular', '700', 'italic']
  subsets   text[],      -- ['latin', 'latin-ext']
  google_fonts_url text,
  preview_url      text,
  embedding vector(256), -- DeepFont feature vector (filled in Phase 3)
  created_at timestamptz default now()
);
```

**`feedback` table — stores user corrections for future training:**

```sql
create table feedback (
  id            serial primary key,
  image_url     text not null,
  predicted_font_id int references fonts(id),
  correct_font_id   int references fonts(id),
  is_correct    boolean,
  created_at    timestamptz default now()
);
```


***

### Step 4 — Create the Vector Search Index

Run this in the SQL Editor to enable fast similarity search:[^3]

```sql
create index if not exists fonts_embedding_idx
on fonts
using ivfflat (embedding vector_cosine_ops)
with (lists = 10);
```

> Note: `lists = 10` is fine for ~1,500 fonts (rule of thumb: `rows / 100`). You can raise this later if you expand beyond 10,000 fonts.[^3]

***

### Step 5 — Create the Vector Search RPC Function

This is a Supabase Postgres function your Next.js API will call to do similarity search. Run in SQL Editor:

```sql
create or replace function match_fonts(
  query_embedding vector(256),
  match_count int default 5
)
returns table (
  id int,
  name text,
  category text,
  google_fonts_url text,
  preview_url text,
  similarity float
)
language sql stable
as $$
  select
    fonts.id,
    fonts.name,
    fonts.category,
    fonts.google_fonts_url,
    fonts.preview_url,
    1 - (fonts.embedding <=> query_embedding) as similarity
  from fonts
  where fonts.embedding is not null
  order by fonts.embedding <=> query_embedding
  limit match_count;
$$;
```


***

### Step 6 — Write the Font Seeding Script

Create `/model/scripts/seed_fonts.py`. This fetches all Google Fonts metadata and inserts it into Supabase — **no embeddings yet**, those come in Phase 3.

```python
import os, requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
GOOGLE_API_KEY = os.environ["GOOGLE_FONTS_API_KEY"]

def fetch_google_fonts():
    url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={GOOGLE_API_KEY}&sort=alpha"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()["items"]

def seed():
    fonts = fetch_google_fonts()
    print(f"Fetched {len(fonts)} fonts from Google Fonts API")

    batch = []
    for font in fonts:
        name = font["family"]
        category = font.get("category", "sans-serif")
        variants = font.get("variants", [])
        subsets = font.get("subsets", [])
        google_fonts_url = f"https://fonts.google.com/specimen/{name.replace(' ', '+')}"

        batch.append({
            "name": name,
            "family": name,
            "category": category,
            "variants": variants,
            "subsets": subsets,
            "google_fonts_url": google_fonts_url,
            "preview_url": None,   # filled later
            "embedding": None      # filled in Phase 3
        })

    # Insert in batches of 100 to avoid timeouts
    for i in range(0, len(batch), 100):
        chunk = batch[i:i+100]
        supabase.table("fonts").insert(chunk).execute()
        print(f"Inserted fonts {i+1} to {i+len(chunk)}")

    print("Seeding complete!")

if __name__ == "__main__":
    seed()
```


***

### Step 7 — Run the Seeding Script

Set up a Python virtual environment inside `/model`:

```bash
cd model
python -m venv venv
source venv/bin/activate        # Mac/Linux
# or: venv\Scripts\activate     # Windows

pip install -r requirements.txt
python scripts/seed_fonts.py
```

You should see output like:

```
Fetched 1521 fonts from Google Fonts API
Inserted fonts 1 to 100
Inserted fonts 101 to 200
...
Seeding complete!
```


***

### Step 8 — Verify in Supabase

1. Go to your Supabase dashboard → **Table Editor → fonts**
2. You should see ~1,500 rows with `name`, `category`, `variants` filled in
3. `embedding` column will be `null` for all rows — that's expected, Phase 3 fills it
4. Run this in SQL Editor to do a quick check:

```sql
select category, count(*) from fonts group by category order by count desc;
```

Expected output:

```
sans-serif   |  650
serif        |  320
display      |  280
handwriting  |  190
monospace    |   81
```


***

### Step 9 — Set Up Row Level Security (RLS)

Since this is a public-facing portfolio app, lock down write access properly:

```sql
-- Enable RLS on both tables
alter table fonts enable row level security;
alter table feedback enable row level security;

-- Anyone can read fonts (for search results)
create policy "Public read fonts"
  on fonts for select using (true);

-- Anyone can insert feedback (for the thumbs up/down button)
create policy "Public insert feedback"
  on feedback for insert with check (true);

-- Only service role can insert/update fonts (for seeding + Phase 3)
create policy "Service role write fonts"
  on fonts for insert with check (auth.role() = 'service_role');

create policy "Service role update fonts"
  on fonts for update using (auth.role() = 'service_role');
```


***

### Step 10 — Commit Your Work

```bash
cd ..  # repo root
git add .
git commit -m "Phase 2: Supabase schema, pgvector setup, RLS policies, font seeding script"
git push origin main
```


***

## Phase 2 Checklist

- [ ] Google Fonts API key created and added to `.env`
- [ ] pgvector extension enabled in Supabase
- [ ] `fonts` table created with `vector(256)` embedding column
- [ ] `feedback` table created for user corrections
- [ ] IVFFlat index created on `embedding` column
- [ ] `match_fonts` RPC function created for similarity search
- [ ] `seed_fonts.py` written and executed successfully
- [ ] ~1,500 font rows visible in Supabase Table Editor
- [ ] RLS policies applied to both tables
- [ ] Changes committed to GitHub

You're now ready to move to **Phase 3: Model Training Pipeline** where you'll synthesize training images, train EfficientNet-B0, and come back to fill in all those `null` embeddings.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://webfontsplugin.com/blog/google-fonts-api-key-complete-wordpress-integration-guide/

[^2]: https://supabase.com/docs/guides/database/extensions/pgvector

[^3]: https://blog.railway.com/p/hosting-postgres-with-pgvector

[^4]: https://developers.google.com/fonts/docs/developer_api

[^5]: https://developers.google.com/fonts/docs/developer_api?apix_params=%7B%22sort%22%3A%22ALPHA%22%7D

[^6]: https://stackoverflow.com/questions/19412940/where-do-i-find-my-google-font-api-key-newbie/20445254

[^7]: https://stackoverflow.com/questions/25652770/how-do-i-acquire-a-google-fonts-api-key/25652856

[^8]: https://www.yo-rent.com/images/pdf/google-fonts-api-setup-guide.pdf

[^9]: https://dev.to/gautam_kumar_d3daad738680/langchain-supabase-vector-store-pgvector-a-beginner-friendly-guide-5h33

[^10]: https://www.youtube.com/watch?v=mdhUvy8PIwA

[^11]: https://www.instaclustr.com/education/vector-database/pgvector-key-features-tutorial-and-pros-and-cons-2026-guide/

[^12]: https://www.kreante.co/post/build-smart-apps-with-supabase-vector-database-semantic-search-guide

[^13]: https://developers.google.cn/fonts/docs/css2?hl=en

[^14]: https://www.milk-island.net/translate/ggd/fonts/docs/developer_api.html

[^15]: https://fontfyi.com/blog/google-fonts-developers-guide/

