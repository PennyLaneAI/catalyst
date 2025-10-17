#!/bin/bash
# 保存為 check_jax_migration.sh

echo "🔍 JAX 0.7.0 遷移檢查"
echo "===================="

echo -e "\n❌ P0 - 必須修改的問題:"
echo "----------------------"

echo -e "\n1. DynamicJaxprTracer 構造函數:"
rg "DynamicJaxprTracer\(" --type py -C 2

echo -e "\n2. trace.getvar() 調用:"
rg "\.getvar\(" --type py

echo -e "\n3. trace.makevar() 調用:"
rg "\.makevar\(" --type py

echo -e "\n4. tracer_to_var 使用:"
rg "tracer_to_var" --type py

echo -e "\n5. frame.tracers 使用:"
rg "frame\.tracers" --type py

echo -e "\n\n⚠️  P1 - 需要檢查的代碼:"
echo "----------------------"

echo -e "\n6. 自定義 staging rules:"
rg "staging_rule|process_primitive" --type py

echo -e "\n7. JaxprStackFrame 使用:"
rg "JaxprStackFrame" --type py

echo -e "\n8. Tracer ID 使用:"
rg "id\(.*tracer" --type py
