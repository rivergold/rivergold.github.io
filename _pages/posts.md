---
layout: archive
permalink: /pages/my_posts
title: "Posts"
author_profile: true
excerpt: "_Stay Hungry, Stay Foolish.<br> A learning route of computer vision and machine learning._"
header:
  overlay_image: /assets/images/home-head.jpg
---

<h2>Posts</h2>
{% for post in site.posts %}
  {% include archive-single.html %}
{% endfor %}
