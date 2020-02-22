---
layout: archive
permalink: /pages/my_posts
title: "Post"
author_profile: true
excerpt: "_Stay hungry. Stay foolish.<br> A record of learning computer vision._"
header:
  overlay_image: /assets/images/home-head.jpg
---

<!-- <h2>Post</h2> -->
{% for post in site.posts %}
  {% include archive-single.html %}
{% endfor %}
