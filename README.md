# nmatrix_reloaded
Reimplementing NMatrix for fun

## NMatrix

[![Build Status](https://travis-ci.org/prasunanand/nmatrix_reloaded.svg?branch=master)](https://travis-ci.org/prasunanand/nmatrix_reloaded)

Fast Numerical Linear Algebra Library for Ruby

## Installation

```sh
git clone https://github.com/prasunanand/nmatrix_reloaded
cd nmatrix_reloaded/
gem install bundler
bundle install
rake compile
```

Run the tests using

```sh
rake test
```

If you want to try out the code without installing:

```sh
rake pry
```

## Speed Test

```
ruby benchmark/bench.rb
```


## Documentation

- http://www.rubydoc.info/github/prasunanand/nmatrix_reloaded

[Yard](https://www.rubydoc.info/gems/yard/) is used for documenting class and methods following yard [tags](https://www.rubydoc.info/gems/yard/file/docs/Tags.md). To generate the static documentation in doc folder run `yard doc`. To serve it in localhost run `yard server`.

# LICENSE

This software is distributed under the [BSD 3-Clause License](LICENSE).

Copyright © 2018-2019, Prasun Anand
