<h1 align="left">Pytorch Live - Custom model</h1>

## :arrow_forward: How to run

1. Follow [PyTorch Live environment setup](https://pytorch.org/live/docs/tutorials/get-started/#prerequisites)
2. Install dependencies and verify setup by running `yarn install`
3. Run Build for either OS

- for iOS
  - run `yarn ios`
- for Android
  - run `yarn android`

## Bump app version

```
yarn bump patch|minor|major
```

It does use `react-native-cli-bump-version` under the hood to bump version name and version code on both platforms.

## Continuous Delivery

We use App Center to to build and deploy applications.
https://appcenter.ms/orgs/Surfrider-Foundation-Europe/applications.

## License

We’re using the `MIT` License. For more details, check [`LICENSE`](https://github.com/surfriderfoundationeurope/App-Plastic-Origins/blob/main/LICENSE) file.
