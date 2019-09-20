NumRuby is part of SciRuby, a collaborative effort to bring scientific computation to Ruby. If you want to help, please do so!

This guide covers ways in which you can contribute to the development of SciRuby and, more specifically, numRuby.

## How to help

There are various ways to help NumRuby: bug reports, coding and documentation. All of them are important.

First, you can help implement new features or bug fixes. To do that, visit our [issue tracker](). If you find something that you want to work on, post it in the issue or on our [mailing list](https://groups.google.com/forum/?fromgroups#!forum/sciruby-dev).

You need to send tests together with your code. No exceptions. You can ask for our opinion, but we won't accept patches without good spec coverage.

We use RSpec for testing. If you aren't familiar with it, there's a good [guide to better specs with RSpec](http://betterspecs.org/) that shows a bit of the syntax and how to use it properly. However, the best resource is probably the specs that already exist -- so just read them.

We only accept bug reports and pull requests in GitHub. You'll need to create a new (free) account if you don't have one already. To learn how to create a pull request, please see [this guide on collaborating](https://help.github.com/categories/63/articles).

If you have a question about how to use NumRuby or SciRuby in general or a feature/change in mind, please ask the [sciruby-dev mailing list](https://groups.google.com/forum/?fromgroups#!forum/sciruby-dev).

Thanks!

## Coding

To start helping with the code, you need to have all the dependencies in place:

- GCC 4.3+
- git
- Ruby 2.3+

Installation:

```bash
$ git clone https://github.com/sciruby/numruby
$ cd numruby/
$ gem install bundler
$ bundle install
$ rake compile
```

Run the tests using

```bash
$ rake test
```

This will install all dependencies, compile the extension and run the specs.

If you want to try out the code without installing:

```bash
$ rake pry
```

Before commiting any code, please read our
[Contributor Agreement](http://github.com/SciRuby/sciruby/wiki/Contributor-Agreement).

## C style guide

TODO: update this section.

* Use snake_case notation for arguments.
* Write a brief description of the arguments that your function receives in the comments directly above the function.
* Explicitly state in the comments any anomalies that your function might have. For example, that it does not work with a certain storage or data type.

## Documentation

- http://www.rubydoc.info/github/prasunanand/nmatrix_reloaded

[Yard](https://www.rubydoc.info/gems/yard/) is used for documenting class and methods following yard [tags](https://www.rubydoc.info/gems yard/file/docs/Tags.md). To generate the static documentation in doc folder run `yard doc`. To serve it in localhost run `yard server`.

## Conclusion

This guide was heavily based on the [Contributing to Ruby on Rails guide](http://edgeguides.rubyonrails.org/contributing_to_ruby_on_rails.html).
