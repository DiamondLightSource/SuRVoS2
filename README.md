# SuRVoS2
Next generation of SuRVoS - Coming soon!

[![pipeline status](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/badges/master/pipeline.svg?job=linux-build&key_text=Linux+build)](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/-/commits/master) [![pipeline status](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/badges/master/pipeline.svg?job=linux-test&key_text=Linux+tests)](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/-/commits/master) [![pipeline status](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/badges/master/pipeline.svg?job=windows-build&key_text=Windows+build&key_width=100)](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/-/commits/master) [![pipeline status](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/badges/master/pipeline.svg?job=windows-test&key_text=Windows+tests&key_width=100)](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/-/commits/master)[![coverage report](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/badges/master/coverage.svg)](https://gitlab.diamond.ac.uk/data-analysis/imaging/SuRVoS2/-/commits/master)

## Run server

```bash
bin/survos server
```

## Run Qt client

```bash
bin/survos qt [workspace_path] --server [host:port]
```

## Run API command

- Locally:

```bash
bin/server run [plugin].[command] [param1=value1] [param2=value2]
```

- Remotely:

```bash
bin/server run [plugin].[command] --server [host:port] [param1=value1] [param2=value2]
```

[![](https://codescene.io/projects/3732/status.svg) Get more details at **codescene.io**.](https://codescene.io/projects/3732/jobs/latest-successful/results)
