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

  gem install --no-document bundler

  sudo apt-get install -y libopenblas-dev
  sudo apt-get install -y liblapack-dev
fi

if [ "$1" = "script" ]
then
  bundle exec rake test
fi
