# toolcall-guard

A [GenericAgent-Harness](https://github.com/Fwind43/GenericAgent-Harness) plugin that catches tool calls a model wrote as plain text and asks it to regenerate, instead of letting the run end.

[GenericAgent-Harness](https://github.com/Fwind43/GenericAgent-Harness) 插件：模型把工具调用写成正文文本时，自动要求它重新生成，而不是让这一轮被当成最终回复结束。

## 问题

部分模型（在 deepseek-v4.1-flash 上观察到）偶尔不发结构化工具调用，而是把调用写进回复正文：

```text
<code_run><parameter name="script">...</parameter></code_run>
<code_run> <｜DSML｜ parameter name="script" string="true">...
<code_run script="...">...</code_run>
```

GA 看不到工具调用，就把这段文本当成最终回复，run 直接结束，用户只能手动发"继续"。

## 行为

- 包装 `GenericAgentHandler.do_no_tool`（"本轮没有工具调用"分支）。不修改 GA 源码，禁用插件时还原。
- 正文命中以下任一写法即拦截，工具名取自 GA 当前的工具表：
  - `<工具名>` 后紧跟 `<parameter ...>`
  - `<工具名>` 后紧跟 DSML 标记
  - `<工具名 参数="...">`
  - `<invoke name="工具名">`
- 命中后给模型回一条"格式无效、未执行任何工具，请用结构化调用重新生成"，并输出一行 `[Warn]`。
- **只检测，不执行**：不会把这段文本解析成真正的调用。
- 连续命中超过 `max_retries` 后按原样放行，避免死循环。

## 设置

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `max_retries` | `2` | 连续要求重新生成的轮数上限（0–5）。`0` 关闭拦截。 |

## 安装

插件面板 → 导入 → Git：

- URL：`https://github.com/Fwind43/gah-plugin-toolcall-guard`
- 分支或 tag：`v0.1.0`

也可以下载源码 zip 后用 zip 导入，或把目录放到 `{data_dir}/plugins/toolcall-guard/` 后重新扫描。启用后从下一次 run 起生效。

## 已知局限

- 围栏代码块和行内代码里的内容不检测，模型把伪调用包在代码块里会漏过。
- 模型在最终回复里裸写 `<code_run>...</code_run>` 当示例会被误拦一次（提示语会让它改用代码块）。
- 依赖 GA 的 `do_no_tool` 方法签名，GA 版本差异可能导致不兼容。

## 状态

v0.1.0 为早期版本：只通过了合成单测（检测样例 + 假 handler 的包装/还原），尚未在真实 harness 运行中验证。启用后请先确认插件状态无报错。

## 插件生态约定

GenericAgent-Harness 的第三方插件仓库建议：

- 打上 GitHub topic [`ga-harness-plugin`](https://github.com/topics/ga-harness-plugin)，便于检索；
- 仓库名用 `gah-plugin-<name>`，`plugin.toml` 放在仓库根目录，可直接用 Git 地址导入；
- 用 `vX.Y.Z` tag 发版，导入时填 tag 可固定版本。
