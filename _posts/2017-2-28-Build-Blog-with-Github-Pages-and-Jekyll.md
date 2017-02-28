# Set up your own blog with Github pages and Jekyll

How can I have a minimalist blog? This article will help you to make it come true.

There are many ways to buid a blog, here we will use Github pages and jekyll to do this. Why? Many programmer use Github pages as their project websites or home pages. Github offer you free server which is very convenient. And it supports Jekyll, a very good static site generator.

## What you need?
- Git
- Jekyll
- shell/powershell
- Your favourite Jekyll theme

## Let's begin
1. Install Git on your computer
    You will need [Git official website][ref_1]


2. Sign up for an Github account
    Please go to [Github official website][ref_2]


3. Create a new repository called _*\<name>\.github.io*_
    <img style="float: center;" src="http://p1.bqimg.com/1949/2704bf4af22b79f8.png" width="300">

4. Install Ruby on you comuter
    You can download from [Ruby official website][ref_3]

5. Install Jekyll
    - Install Jekyll on Windows[(ref)][ref_4]:
        - use Chocolatey install ruby `choco install ruby -y [-v Version]`

        - `gem install bundler` If it shows there is no gem, you should close current terminal and open it again(restart terminal)

        - Install bundler(a manager of ruby applications) via 'gem install bundler'

        - Install Jekyll via `gem install jekyll`

        - Problem and Solution
        - If `SSL_connect returned=1 errno=0 state=SSLv3 read server certificate B: certificate verify failed` error occurs, you should update gem by download the latest rubygem from [here](https://rubygems.org/pages/download?locale=zh-CN#formats).

6. Create blog with Jekyll:
- One way, you can build a new blog folder by `jekyll new <file name>`
- Another way, you can using jekyll themes and templates online, here are some good themes which are minimalistic :
    - [Minimal Mistakes Jekyll Theme](https://github.com/mmistakes/minimal-mistakes)
    - [chalk](https://github.com/nielsenramon/chalk)
    - [indigo](https://github.com/sergiokopplin/indigo)
    - [The Plain](http://jekyllthemes.org/themes/the-plain/)
    - [Zetsu](http://jekyllthemes.org/themes/zetsu/)
By the way, my blog is built based on _Minimal Mistakes Jekyll Theme_ which is created by [mmistakes](https://github.com/mmistakes). I love this minimalistic  theme and it is very easy to install with `ruby gem`.

7. Jekyll base tutorials:
    - What is Jekyll or how to understand it?
    In my view, it is a framework offer you website templates and can convert .md into html. One of the most important concepts and tools of Jekyll is __Layout__, which defines and provides html templates to you. Another important thing is __YAML Front Matter__, a config command just like _yaml_. The __YAML Front Matter__ must be the first thing in the file and must take the form of valid YAML set between triple-dashed lines. Here is a basic example:
    ```
        ---
        layout: post
        title: Blogging Like a Hacker
        ---
    ```

    - Here are common command of Jekyll:
     - `jekyll new <file name>`
     Create a new blog file

     - `jekyll build`
     Build your blog.

     - `bundle exec jekyll serve`
     Use local serve to preview your blog

     For more details, you should browse [Jekyll Docs](https://jekyllrb.com/docs/home/).

Wish you will build your favurite blog by Github Pages and Jekyll :relaxed:.

[ref_1]:https://git-scm.com/
[ref_2]:https://github.com/
[ref_3]:https://www.ruby-lang.org/en/downloads/
[ref_4]:https://jekyllrb.com/docs/windows/
