---
title: 使用Hexo搭建博客
date: 2022-03-01 17:53:46
categories: Tech
tags:
  - Hexo
  - JavaScript
---

之前是使用jekyll搭建的博客，但是总有些不太满意的地方，也因为自己对Ruby不了解，所以决心将博客迁移至Hexo进行搭建，这里记录下自己的一些经验。

## Hexo基础命令

### 创建草稿

```bash
hexo new draft <title>
```

### 发布草稿

```bash
hexo publish post <title>
```

### 本地预览

```bash
hexo clean && hexo s -p <端口号> --draft
```

- `--draft`: 对`_drafts`下的文章也进行预览

### 部署github pages

首先，需要在_config.yaml进行部署相关的配置:

```yaml
# Deployment
## Docs: https://hexo.io/docs/one-command-deployment
deploy:
  type: git
  repo: <Github仓库ssh链接> # example, git@github.com:xxx/xxx.github.io.git
  branch: <分支名> # example, feature/hexo
# # Markdown-it config
# ## Docs: https://github.com/celsomiranda/hexo-renderer-markdown-it/wiki
# ## cn_Docs: https://markdown-it.docschina.org/api/MarkdownIt.html#markdownit-new
```

之后，需要在Github仓库中的Github Pages配置页面，对source中的Branch进行配置，需要和`_config.yaml`中的`branch`保持一致。

然后，待本地编写完文章内容并预览ok后，执行以下命令，hexo会进行编译并将生成的网站内容上传至Github仓库中

```bash
hexo clean && hexo deploy
```

最后，坐等一会刷新下你的网站，文章就展示出来咯～

---

## Hexo Markdown

### 引用站内文章

```markdown
{% post_link <markdown_file_name> <show_name> %}
```

- `markdown_file_name`: 需要引用的文章在_posts目录下的文件名
- `show_name`: 引用链接现实的内容

---

## 主题配置

在浏览和对比了一众主题后，rivergold最终选择了Butterfly作为基础主题。该主题风格偏简约且配置功能相对完善，比较适合个人的风格。

### 安装主题

根据Butterfly官方文档，安装方式主要有两种

{% tabs 安装主题 %}
<!-- tab npm -->
```bash
npm i hexo-theme-butterfly
```
<!-- endtab -->
<!-- tab git submodule-->
```bash
git submodule add https://github.com/jerryc127/hexo-theme-butterfly.git themes/butterfly
```
<!-- endtab -->
{% endtabs %}

为了能获取最新的Butterfly的功能，同时跟踪作者的修改，rivergold采用的是将Butterfly作为submodule进行管理。对`git submodule`不熟的盆友可以参考{% post_link git_memo-使用submodule Git Memo - 使用submodule %}

### 修改主页subtitle的打字动画速度

由于主题默认的配置中没有调整打字动画速度的参数，所以才用了修改源码这种简单直接的方法。。。
源码位置在`butterfly/layout/include/third-party/subtitle.pug`中，控制打字速度的变量为`typeSpeed`，原始设置为150，数值越大打字动画速度越慢（这个数值应该表示的停留间隙）。

```pug
case source
  when 1
    script.
      function subtitleType () {
        fetch('https://v1.hitokoto.cn')
          .then(response => response.json())
          .then(data => {
            if (!{effect}) {
              const from = '出自 ' + data.from
              const sub = !{JSON.stringify(subContent)}
              sub.unshift(data.hitokoto, from)
              window.typed = new Typed('#subtitle', {
                strings: sub,
                startDelay: 300,
                typeSpeed: 100,
                loop: !{loop},
                backSpeed: 100,
              })
            } else {
              document.getElementById('subtitle').innerHTML = data.hitokoto
            }
          })
      }
```

### 配置锚点

由于Butterfly默认的锚点配置相对简单且不能控制锚点显示层级和自定义更好的锚点图标，因此rivergold走上了自定义锚点配置的探索之路。参考了众多博客后，发现替换hexo的markdown渲染器并增加相关配置后可以解决问题。

Hexo默认的markdown渲染器为`hexo-renderer-marked`，其支持的相关功能有限；这里建议替换为[``hexo-renderer-markdown-it``](https://github.com/hexojs/`hexo-renderer-markdown-it`)：

```bash
# 在博客根目录下执行命令卸载hexo-renderer-marked
npm un hexo-renderer-marked --save
# 安装`hexo-renderer-markdown-it`
npm i `hexo-renderer-markdown-it` --save
```

#### 配置``hexo-renderer-markdown-it``

修改`_config.yml`，添加如下配置：

```yaml
markdown:
  render:
    # false。设成 true 来启用在源码中(支持) HTML 标签。注意！这是不安全的！你可能需要额外的消毒剂(sanitizer)来组织来自 XSS 的输出。最好是通过插件来扩展特性，而不是启用 HTML。
    html: true
    # 设成 true 来给闭合的单个标签（<br />）添加 '/'。只有完全兼容 CommonMark 模式时才需要这样做。实际上你只需要 HTML 输出。
    xhtmlOut: false
    # 设成 true 来转化段落里的 \n 成 <br>
    breaks: true
    # 设成 true 来自动转化像 URL 的文本成链接。
    linkify: true
    # 设成 true 来启用某些语言中性的替换以及引号的美化（智能引号）。
    typographer: true
    # String 或 Array 类型。在 typographer 启用和支持智能引号时，进行双引号 + 单引号对替换。 比方说，
    # 你可以支持 '«»„“' 给俄罗斯人使用， '„“‚‘' 给德国人使用。
    # 还有 ['«\xA0', '\xA0»', '‹\xA0', '\xA0›']  给法国人使用（包括 nbsp）。
    quotes: "“”‘’"
  plugins:
    - markdown-it-abbr
    - markdown-it-footnote
    - markdown-it-ins
    - markdown-it-sub
    - markdown-it-sup
    - markdown-it-emoji
  anchors:
    # >=level的标题会添加permalink
    level: 2
    collisionSuffix: "v"
    # If `true`, creates an anchor tag with a permalink besides the heading.
    # 如果为“true”，则在标题旁边创建一个带有永久链接的定位标记。
    permalink: true
    permalinkClass: header-anchor
    # The symbol used to make the permalink
    # 用于生成永久链接的符号，支持emoji
    permalinkSymbol: ":cactus: "
    # 设定链接图标在标题left还是right
    permalinkSide: "left"
    # 转换锚点 ID 中的字母为大写或小写 # "0" 不转换, "1" 为小写, "2" 为大写
    case: 0
    # 用于替换空格的符号, 默认为 "-"
    separator: "-"
```

其中，需要注意的配置信息为：

```yaml
plugins:
    - markdown-it-emoji
```

该插件使得markdown支持emoji表情，可以使得你的博客标题更好看哦～

[`hexo-renderer-markdown-it` 的配置与插件配置](https://blog.bugimg.com/works/`hexo-renderer-markdown-it`_and_plugins_config.html)
[hexo - 使用渲染器 - `hexo-renderer-markdown-it`](https://lamirs.vercel.app/hexo-%E4%BD%BF%E7%94%A8%E6%B8%B2%E6%9F%93%E5%99%A8-`hexo-renderer-markdown-it`/)

另外还有一点需要注意，对于`anchors`中的`level`配置，指的是>=`level`的所有标题都会添加permalink并进行渲染。原始的`hexo-renderer-markdown-it`并不支持配置从某个`level`的header到另一个level的header的permalink，这会导致文章中几乎所有的标题前面都加上的`permalinkSymbol`图标，rivergold个人觉得有点太繁杂了，所以对`hexo-renderer-markdown-it`中对anchors配置的源码进行了修改（虽然我是做算法的，但也能照猫画虎改js代码。。。）

`hexo-renderer-markdown-it`中对anchors配置的源代码在`node_modules/hexo-renderer-markdown-it`/lib/anchors.js`

```javascript
const anchor = function (md, opts) {
    Object.assign(opts, { renderPermalink });

    const titleStore = {};
    const originalHeadingOpen = md.renderer.rules.heading_open;
    const slugOpts = { transform: opts.case, ...opts };

    md.renderer.rules.heading_open = function (...args) {
        const [tokens, idx, something, somethingelse, self] = args; // eslint-disable-line no-unused-vars

        if (tokens[idx].tag.substr(1) >= opts.level) {
            let _tokens$idx;

            const title = tokens[idx + 1].children.reduce((acc, t) => {
                return acc + t.content;
            }, '');

            let slug = slugize(title, slugOpts);

            if (Object.prototype.hasOwnProperty.call(titleStore, slug)) {
                titleStore[slug] = titleStore[slug] + 1;
                slug = slug + '-' + opts.collisionSuffix + titleStore[slug].toString();
            } else {
                titleStore[slug] = 1;
            }


            (_tokens$idx = tokens[idx], !_tokens$idx.attrs && (_tokens$idx.attrs = []), _tokens$idx.attrs)
                .push(['id', slug]);

            // 修改：只对level-1的header进行渲染
            if (tokens[idx].tag.substr(1) == opts.level) {
                if (opts.permalink) {
                    opts.renderPermalink.apply(opts, [slug, opts].concat(args));
                }
            }

        }

        return originalHeadingOpen
            ? originalHeadingOpen.apply(this, args)
            : self.renderToken.apply(self, args);
    };
};
```

有人在`hexo-renderer-markdown-it`仓库中提出了类似问题的[issue#177](https://github.com/hexojs/hexo-renderer-markdown-it/issues/177)，rivergold也在进行了回复，虽然没有提交PR但还是希望可以帮助到大家:grinning:。

那如果实现不同level的header使用不同的permalinkSymbol呢？实现也很简单，参考`renderPermalink`函数新增加`renderSubPermalink`函数和`subPermalinkSymbol`变量便可以实现：

```javascript
// 用于渲染level-2以下的header
const renderSubPermalink = function (slug, opts, tokens, idx) {
    const permalink = [Object.assign(new Token('link_open', 'a', 1), {
        attrs: [['class', opts.permalinkClass], ['href', '#' + slug]]
    }), Object.assign(new Token('text', '', 0), {
        content: opts.subPermalinkSymbol
    }), new Token('link_close', 'a', -1), Object.assign(new Token('text', '', 0), {
        content: ''
    })];

    if (opts.permalinkSide === 'right') {
        return tokens[idx + 1].children.push(...permalink);
    }

    return tokens[idx + 1].children.unshift(...permalink);
};
```

```javascript
// 添加逻辑控制
if (tokens[idx].tag.substr(1) == opts.level) {
    if (opts.permalink) {
        opts.renderPermalink.apply(opts, [slug, opts].concat(args));
    }
}
else{
    if (opts.permalink) {
        opts.renderSubPermalink.apply(opts, [slug, opts].concat(args));
    }
}
```

修改后完整的`anchors.js`已同步到了[GitHubGist](https://gist.github.com/rivergold/bc7af863a74ffaf2d3d0109b768c6ad8)，需要的盆友可以参考哦～

<!-- ### 添加sitemap

为了能让Google和Baidu收录自己的博客，让大家可以在搜索引擎中检索到，我们需要对自己的博客网站进行SEO优化。

#### 使用hexo-generator-sitemap生成sitemap.xml

```bash
npm install hexo-generator-sitemap --save
```

[让Google搜索到自己的博客](https://zoharandroid.github.io/2019-08-03-%E8%AE%A9%E8%B0%B7%E6%AD%8C%E6%90%9C%E7%B4%A2%E5%88%B0%E8%87%AA%E5%B7%B1%E7%9A%84%E5%8D%9A%E5%AE%A2/)
[Hexo sitemap插件](https://butterfly.js.org/posts/4073eda/#%E6%8F%92%E4%BB%B6%E6%8E%A8%E8%96%A6) -->

References:

- [Hexo文档](https://hexo.io/zh-tw/docs/index.html)
- [Butterfly主题](https://butterfly.js.org/)
