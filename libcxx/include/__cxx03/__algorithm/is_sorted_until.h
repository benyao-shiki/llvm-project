//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_IS_SORTED_UNTIL_H
#define _LIBCPP___CXX03___ALGORITHM_IS_SORTED_UNTIL_H

#include <__cxx03/__algorithm/comp.h>
#include <__cxx03/__algorithm/comp_ref_type.h>
#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Compare, class _ForwardIterator>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
__is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp) {
  if (__first != __last) {
    _ForwardIterator __i = __first;
    while (++__i != __last) {
      if (__comp(*__i, *__first))
        return __i;
      __first = __i;
    }
  }
  return __last;
}

template <class _ForwardIterator, class _Compare>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp) {
  return std::__is_sorted_until<__comp_ref_type<_Compare> >(__first, __last, __comp);
}

template <class _ForwardIterator>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last) {
  return std::is_sorted_until(__first, __last, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ALGORITHM_IS_SORTED_UNTIL_H
