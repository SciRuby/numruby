#!/bin/bash


if [ "$1" = "install" ]
then
  bundle install
  bundle exec rake compile
fi

if [ "$1" = "before_install" ]
then
  case "$TRAVIS_OS_NAME" in
    linux)
      sudo apt-get update -qq
      ;;
  esac

  if [ $TRAVIS_RUBY_VERSION == '2.2.4' ] || [ $TRAVIS_RUBY_VERSION == '2.1.8' ]
  then
    gem install --no-document bundler -v '~> 1.6';
    gem install --no-document parallel -v '1.13.0';
  else
    gem install --no-document bundler;
  fi

  sudo apt-get install -y libopenblas-dev
  sudo apt-get install -y liblapack-dev
  sudo apt-get install -y liblapacke-dev
fi

if [ "$1" = "script" ]
then
  bundle exec rake test
fi
