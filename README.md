# Competitive control systems workshop

## Instructions

Come up with a creative <team_name>.

Get started with:

```shell
curl -H "Authorization: Bearer <team_name>" -X POST http://157.230.103.230:10300/challenge/drone
```

Or using `httpie`:

```shell
http -v POST http://157.230.103.230:10300/challenge/drone -A bearer --auth <team_name>
```

Receive further instructions:

```shell
curl -H "Authorization: Bearer firstname.lastname" -X GET http://157.230.103.230:10300/challenge/drone
```

Or using `httpie`:

```shell
http -v GET http://157.230.103.230:10300/challenge/drone -A bearer --auth <team_name>
```

Then, use your preferred language and go!

## Using the template (optional)

Beyond the instructions above, you can also clone this repository and use it as a starter template for your project. It's a `nix`-powered Rust project that should save you some time setting things up.

> [!TIP]
> Using this template is completely optional. If you want to start from scratch and fight dependencies yourself, go for it!
