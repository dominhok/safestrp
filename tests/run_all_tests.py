#!/usr/bin/env python3
"""
🚀 SafeStrp 전체 테스트 실행기

모든 개별 테스트들을 순차적으로 실행하고
종합적인 결과를 제공합니다.
"""

import subprocess
import sys
import time
import traceback
from datetime import datetime

def run_test_script(script_name, description):
    """개별 테스트 스크립트 실행"""
    print(f"\n{'='*80}")
    print(f"🔍 {description}")
    print(f"   실행 파일: {script_name}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        
        # 테스트 실행
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 결과 출력
        if result.returncode == 0:
            print("✅ 테스트 실행 성공")
            print(f"⏱️  실행 시간: {execution_time:.2f}초")
        else:
            print("❌ 테스트 실행 실패")
            print(f"⏱️  실행 시간: {execution_time:.2f}초")
            print(f"❌ 오류 코드: {result.returncode}")
        
        # 출력 내용 표시 (마지막 30줄만)
        if result.stdout:
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 30:
                print("\n📋 출력 결과 (마지막 30줄):")
                for line in output_lines[-30:]:
                    print(f"   {line}")
            else:
                print("\n📋 출력 결과:")
                for line in output_lines:
                    print(f"   {line}")
        
        # 에러 출력
        if result.stderr:
            print("\n❌ 에러 출력:")
            error_lines = result.stderr.strip().split('\n')
            for line in error_lines[-10:]:  # 마지막 10줄만
                print(f"   {line}")
        
        return {
            'script': script_name,
            'description': description,
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("⏰ 테스트 타임아웃 (5분 초과)")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'execution_time': 300,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"💥 예상치 못한 오류: {e}")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'execution_time': 0,
            'error': str(e)
        }

def extract_test_summary(output):
    """테스트 출력에서 요약 정보 추출"""
    if not output:
        return {}
    
    summary = {}
    lines = output.split('\n')
    
    # 성공률 찾기
    for line in lines:
        if '성공률:' in line:
            try:
                rate = line.split('성공률:')[1].split('%')[0].strip()
                summary['success_rate'] = float(rate)
            except:
                pass
        
        if '총 테스트:' in line:
            try:
                total = line.split('총 테스트:')[1].split('개')[0].strip()
                summary['total_tests'] = int(total)
            except:
                pass
        
        if '성공:' in line and '개' in line:
            try:
                success = line.split('성공:')[1].split('개')[0].strip()
                summary['successful_tests'] = int(success)
            except:
                pass
        
        if 'FPS:' in line:
            try:
                fps = line.split('FPS:')[1].split()[0].strip()
                summary['max_fps'] = float(fps)
            except:
                pass
    
    return summary

def print_final_summary(test_results):
    """최종 종합 요약 출력"""
    print("\n" + "="*80)
    print("🎯 SafeStrp 전체 테스트 종합 결과")
    print("="*80)
    
    total_scripts = len(test_results)
    successful_scripts = sum(1 for r in test_results if r['success'])
    
    print(f"\n📊 전체 스크립트 실행 결과:")
    print(f"   총 테스트 스크립트: {total_scripts}개")
    print(f"   성공한 스크립트: {successful_scripts}개")
    print(f"   실패한 스크립트: {total_scripts - successful_scripts}개")
    print(f"   스크립트 성공률: {(successful_scripts/total_scripts)*100:.1f}%")
    
    # 개별 스크립트 결과
    print(f"\n📋 개별 스크립트 결과:")
    total_execution_time = 0
    
    for result in test_results:
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['execution_time']:.1f}초"
        print(f"   {status} {result['description']}")
        print(f"       파일: {result['script']}")
        print(f"       실행시간: {time_str}")
        
        if 'error' in result:
            print(f"       오류: {result['error']}")
        
        # 요약 정보 추출
        if result['success'] and 'stdout' in result:
            summary = extract_test_summary(result['stdout'])
            if summary:
                if 'success_rate' in summary:
                    print(f"       성공률: {summary['success_rate']:.1f}%")
                if 'total_tests' in summary:
                    print(f"       테스트 수: {summary['total_tests']}개")
                if 'max_fps' in summary:
                    print(f"       최고 FPS: {summary['max_fps']:.1f}")
        
        total_execution_time += result['execution_time']
        print()
    
    print(f"📊 전체 실행 시간: {total_execution_time:.1f}초 ({total_execution_time/60:.1f}분)")
    
    # 최종 평가
    print(f"\n🎯 최종 평가:")
    
    if successful_scripts == total_scripts:
        print("   ✅ 모든 테스트 스크립트가 성공적으로 실행되었습니다!")
        print("   ✅ SafeStrp 코드베이스가 양호한 상태입니다.")
    elif successful_scripts >= total_scripts * 0.8:
        print("   ⚠️  대부분의 테스트가 성공했지만 일부 문제가 있습니다.")
        print("   📝 실패한 테스트들을 개별적으로 검토해보세요.")
    else:
        print("   ❌ 여러 테스트에서 문제가 발견되었습니다.")
        print("   🔧 코드베이스 상태 점검이 필요합니다.")
    
    # 권장사항
    failed_scripts = [r for r in test_results if not r['success']]
    if failed_scripts:
        print(f"\n💡 권장사항:")
        print(f"   • 실패한 스크립트들을 개별적으로 실행해보세요:")
        for failed in failed_scripts:
            print(f"     - python {failed['script']}")
        
        if any('timeout' in r.get('error', '').lower() for r in failed_scripts):
            print(f"   • 타임아웃이 발생한 경우 더 작은 배치로 테스트해보세요.")
        
        if any('import' in r.get('stderr', '').lower() for r in failed_scripts):
            print(f"   • Import 오류가 있는 경우 필요한 패키지를 설치하세요.")
    
    print("\n" + "="*80)

def main():
    """메인 실행 함수"""
    print("🚀 SafeStrp 전체 테스트 실행기 시작")
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 실행할 테스트 스크립트들 (순서대로)
    test_scripts = [
        {
            'script': 'test_1_imports_dependencies.py',
            'description': 'Import 및 의존성 테스트'
        },
        {
            'script': 'test_2_model_architecture.py',
            'description': '모델 구조 테스트'
        },
        {
            'script': 'test_3_loss_functions.py',
            'description': '손실 함수 테스트'
        },
        {
            'script': 'test_4_inference_pipeline.py',
            'description': '추론 파이프라인 테스트'
        },
        {
            'script': 'test_5_cross_task_features.py',
            'description': 'Cross-task Consistency 테스트'
        }
    ]
    
    test_results = []
    
    try:
        # 각 테스트 스크립트 실행
        for i, test_config in enumerate(test_scripts, 1):
            print(f"\n🔄 진행률: {i}/{len(test_scripts)}")
            
            result = run_test_script(
                test_config['script'],
                test_config['description']
            )
            test_results.append(result)
            
            # 실패한 경우에도 계속 진행
            if not result['success']:
                print(f"⚠️  {test_config['description']} 실패했지만 다음 테스트 계속 진행...")
            
            # 메모리 정리를 위한 잠시 대기
            time.sleep(2)
        
        # 최종 종합 요약
        print_final_summary(test_results)
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 테스트가 중단되었습니다.")
        if test_results:
            print("⚠️  현재까지의 결과를 요약합니다...")
            print_final_summary(test_results)
    except Exception as e:
        print(f"\n💥 예상치 못한 오류 발생: {e}")
        traceback.print_exc()
        if test_results:
            print("⚠️  현재까지의 결과를 요약합니다...")
            print_final_summary(test_results)

if __name__ == "__main__":
    main() 