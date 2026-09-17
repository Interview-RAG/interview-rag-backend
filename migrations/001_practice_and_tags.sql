-- PrepAI — Modernist redesign migration
-- Adds tagging + spaced repetition to qa_records, and a review history table
-- that powers the Practice and Progress screens.
--
-- Safe to run more than once. Run this in the Supabase SQL editor before
-- starting the new backend.

-- ── 1. qa_records: tags + spaced-repetition state ────────────────────────────
alter table qa_records add column if not exists tags          text[]      not null default '{}';
alter table qa_records add column if not exists due_date      date        not null default current_date;
alter table qa_records add column if not exists review_count  integer     not null default 0;
alter table qa_records add column if not exists last_ease     text;
alter table qa_records add column if not exists interval_days integer     not null default 0;
alter table qa_records add column if not exists created_at    timestamptz not null default now();

alter table qa_records drop constraint if exists qa_records_last_ease_check;
alter table qa_records add  constraint qa_records_last_ease_check
  check (last_ease is null or last_ease in ('again', 'good', 'easy'));

create index if not exists qa_records_user_due_idx  on qa_records (user_id, due_date);
create index if not exists qa_records_tags_gin_idx  on qa_records using gin (tags);

-- ── 2. qa_reviews: one row per rating, for streaks and the 14-day chart ──────
-- user_id is created with the same type qa_records.user_id already uses, so
-- this works whether that column is uuid, text or bigint.
do $$
declare
  uid_type text;
begin
  select format_type(a.atttypid, a.atttypmod)
    into uid_type
    from pg_attribute a
   where a.attrelid = 'qa_records'::regclass
     and a.attname  = 'user_id'
     and a.attnum   > 0;

  if uid_type is null then
    raise exception 'qa_records.user_id not found — is this the right database?';
  end if;

  execute format($fmt$
    create table if not exists qa_reviews (
      id          bigserial primary key,
      qa_id       bigint      not null references qa_records(id) on delete cascade,
      user_id     %s          not null,
      ease        text        not null check (ease in ('again', 'good', 'easy')),
      score       integer     check (score between 0 and 10),
      mode        text        not null default 'flip' check (mode in ('flip', 'type')),
      reviewed_at timestamptz not null default now()
    )
  $fmt$, uid_type);
end $$;

create index if not exists qa_reviews_user_time_idx on qa_reviews (user_id, reviewed_at desc);
create index if not exists qa_reviews_qa_idx        on qa_reviews (qa_id);

-- ── 3. Backfill: existing pairs are due today and untagged ──────────────────
update qa_records
   set due_date = current_date
 where due_date is null;
