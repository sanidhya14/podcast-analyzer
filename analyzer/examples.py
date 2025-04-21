TRANSCRIPT_ANALYSIS_PROMPT_EXAMPLES = """
### Example 1
Transcript:
(H) - [00:00:01]: Welcome to another episode of Culture Talks. Today we have Anjali Sharma with us.
(G) - [00:00:10]: Thank you, H! Super excited to be here.
(H) - [00:00:15]: So, you grew up in Jaipur right?
(G) - [00:00:18]: Haan, Jaipur. But the heat was something else. May ke mahine mein, it felt like oven khul gaya ho.

Output:
{
  "chapters": [
    {
      "start_timestamp": "00:00:01",
      "chapter_title": "Introducing Anjali and Her Jaipur Roots"
    }
  ],
  "clip_worthy_moments": [
    {
      "type": "relatable",
      "timestamp": "00:00:18",
      "quote": "May ke mahine mein, it felt like oven khul gaya ho.",
      "why_clipworthy": "A humorous and highly relatable description of Indian summer."
    }
  ]
}

### Example 2
Transcript:
(G) - [00:04:50]: Honestly, I was completely lost during my second year of college.
(G) - [00:04:58]: I thought engineering was my dream, but turns out it was my dad's dream.
(H) - [00:05:03]: That must have been tough.

Output:
{
  "chapters": [
    {
      "start_timestamp": "00:04:50",
      "chapter_title": "Struggles of Finding One’s Own Path"
    }
  ],
  "clip_worthy_moments": [
    {
      "type": "vulnerable",
      "timestamp": "00:04:58",
      "quote": "I thought engineering was my dream, but turns out it was my dad's dream.",
      "why_clipworthy": "Emotionally honest and likely to resonate with many listeners facing parental pressure."
    }
  ]
}

### Example 3
Transcript:
(H) - [00:10:12]: What’s the craziest food you’ve tried?
(G) - [00:10:15]: Bhai, Vietnam mein ek baar kuch aaya jisme tentacles hil rahe the. And I still ate it.
(H) - [00:10:21]: You’re braver than me!

Output:
{
  "chapters": [
    {
      "start_timestamp": "00:10:12",
      "chapter_title": "Adventures in Vietnamese Street Food"
    }
  ],
  "clip_worthy_moments": [
    {
      "type": "funny",
      "timestamp": "00:10:15",
      "quote": "Vietnam mein ek baar kuch aaya jisme tentacles hil rahe the.",
      "why_clipworthy": "An amusing and vivid travel anecdote that creates a memorable visual."
    }
  ]
}
"""
