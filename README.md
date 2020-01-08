# NumRuby

[![Build Status](https://travis-ci.org/sciruby/numruby.svg?branch=master)](https://travis-ci.org/sciruby/numruby)

Fast Numerical Linear Algebra Library for Ruby. NMatrix reimplementation.

## Installation

### Install dependency libraries

**Debian/Ubuntu**

```sh
sudo apt-get install libopenblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install liblapacke-dev
```

**Mac OSX**

```sh
brew install openblas
brew install lapack
```

### Build the library

```sh
git clone https://github.com/sciruby/numruby
cd numruby/
gem install bundler
bundle install
rake compile
```

### Run tests

```sh
rake test
```

### Try out the code without installing

```sh
rake pry
```

## Speed Test

```sh
ruby benchmark/bench.rb
```

## Documentation

- http://www.rubydoc.info/github/prasunanand/nmatrix_reloaded

[Yard](https://www.rubydoc.info/gems/yard/) is used for documenting class and methods following yard [tags](https://www.rubydoc.info/gems/yard/file/docs/Tags.md). To generate the static documentation in doc folder run `yard doc`. To serve it in localhost run `yard server`.

## LICENSE

This software is distributed under the [BSD 3-Clause License](LICENSE).

Copyright © 2018-2019, Prasun Anand
