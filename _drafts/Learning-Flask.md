# Basics

## `routes`

> So what goes in the routes module? The routes are the different URLs that the application implements. In Flask, handlers for the application routes are written as Python functions, called view functions. View functions are mapped to one or more route URLs so that Flask knows what logic to execute when a client requests a given URL.

> If you could keep the logic of your application separate from the layout or presentation of your web pages, then things would be much better organized

> The `render_template()` function invokes the `Jinja2` template engine that comes bundled with the Flask framework. Jinja2 substitutes {{ ... }} blocks with the corresponding values, given by the arguments provided in the render_template() call.

## Blueprints

In Flask, a blueprint is a logical structure that represents a subset of the application. A blueprint can include elements such as routes, view functions, forms, templates and static files. If you write your blueprint in a separate Python package, then you have a component that encapsulates the elements related to specific feature of the application.

**So you can think of a blueprint as a temporary storage for application functionality that helps in organizing your code.**

# HTML

```html
<html>
    <head>
        <title>{{ title }} - Microblog</title>
    </head>
    <body>
        <h1>Hello, {{ user.username }}!</h1>
    </body>
</html>
```

`{{ ... }}` is a placeholders for the dynamic content, which represent the parts of the page that are variable and will only be known at runtime.