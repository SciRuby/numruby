# coding: utf-8
$:.unshift File.expand_path("../lib", __FILE__)

require 'nmatrix/version.rb'

Gem::Specification.new do |gem|
  gem.name        = 'nmatrix'
  gem.version     = '0.0.1'
  gem.date        = '2018-09-29'
  gem.summary     = 'Linear Algebra'
  # gem.description =  NMatrix::DESCRIPTION
  gem.authors     = ['Prasun Anand']
  gem.email       = 'prasunanand.bitsp@gmail.com'
  gem.files       = ['lib/nmatrix.rb']
  gem.homepage    = 'http://rubygems.org/gems/nmatrix'
  gem.license     = 'BSD-3-Clause'
  # gem.extensions  = ['ext/extconf.rb']
  gem.files       = `git ls-files -- ext/`.split("\n")
  gem.files      += `git ls-files -- lib`.split("\n")
  gem.require_paths = ["lib"]
  gem.add_development_dependency 'rubocop', '~> 0'
  gem.add_development_dependency 'bundler', '~>1.6'
  gem.add_development_dependency 'json', '~> 0'
  gem.add_development_dependency 'pry', '~>0.10'
  gem.add_development_dependency 'rake', '~>10.3'
  gem.add_development_dependency 'rake-compiler', '~>0.8'
  gem.add_development_dependency 'rdoc', '~>4.0', '>=4.0.1'
  gem.add_development_dependency "minitest", "~> 5.0"
end
