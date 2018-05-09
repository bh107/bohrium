lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "bohrium/version"

Gem::Specification.new do |spec|
  spec.name          = "bohrium"
  spec.version       = Bohrium::VERSION
  spec.authors       = ["Mads Ohm Larsen"]
  spec.email         = ["mads.ohm@gmail.com"]

  spec.summary       = %q{Speed up matrix computations.}
  spec.description   = %q{Speed up your matrix computations by auto generating OpenMP/OpenCL kernels via Bohrium.}
  spec.homepage      = "https://bohrium.readthedocs.io"
  spec.license       = "GPL-3.0"

  spec.files         = ["Gemfile", "README.md", "Rakefile", Dir["lib/**/*.rb"], Dir["ext/**/*"], "bohrium.gemspec", Dir["spec/**/*"]]
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/bohrium/extconf.rb", "ext/bohrium/templates"]

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rspec", "~> 3.7"
  spec.add_development_dependency "rake-compiler", "~> 1.0"
end
