// Copyright (c) 2018 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_UINT8_HPP
#define TAO_PEGTL_UINT8_HPP

#include "config.hpp"

#include "internal/peek_mask_uint8.hpp"
#include "internal/peek_uint8.hpp"
#include "internal/result_on_found.hpp"
#include "internal/rules.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace uint8
      {
         // clang-format off
         struct any : internal::any< internal::peek_uint8 > {};

         template< std::uint8_t... Cs > struct not_one : internal::one< internal::result_on_found::FAILURE, internal::peek_uint8, Cs... > {};
         template< std::uint8_t Lo, std::uint8_t Hi > struct not_range : internal::range< internal::result_on_found::FAILURE, internal::peek_uint8, Lo, Hi > {};
         template< std::uint8_t... Cs > struct one : internal::one< internal::result_on_found::SUCCESS, internal::peek_uint8, Cs... > {};
         template< std::uint8_t Lo, std::uint8_t Hi > struct range : internal::range< internal::result_on_found::SUCCESS, internal::peek_uint8, Lo, Hi > {};
         template< std::uint8_t... Cs > struct ranges : internal::ranges< internal::peek_uint8, Cs... > {};
         template< std::uint8_t... Cs > struct string : internal::seq< internal::one< internal::result_on_found::SUCCESS, internal::peek_uint8, Cs >... > {};

         template< std::uint8_t M, std::uint8_t... Cs > struct mask_not_one : internal::one< internal::result_on_found::FAILURE, internal::peek_mask_uint8< M >, Cs... > {};
         template< std::uint8_t M, std::uint8_t Lo, std::uint8_t Hi > struct mask_not_range : internal::range< internal::result_on_found::FAILURE, internal::peek_mask_uint8< M >, Lo, Hi > {};
         template< std::uint8_t M, std::uint8_t... Cs > struct mask_one : internal::one< internal::result_on_found::SUCCESS, internal::peek_mask_uint8< M >, Cs... > {};
         template< std::uint8_t M, std::uint8_t Lo, std::uint8_t Hi > struct mask_range : internal::range< internal::result_on_found::SUCCESS, internal::peek_mask_uint8< M >, Lo, Hi > {};
         template< std::uint8_t M, std::uint8_t... Cs > struct mask_ranges : internal::ranges< internal::peek_mask_uint8< M >, Cs... > {};
         template< std::uint8_t M, std::uint8_t... Cs > struct mask_string : internal::seq< internal::one< internal::result_on_found::SUCCESS, internal::peek_mask_uint8< M >, Cs >... > {};
         // clang-format on

      }  // namespace uint8

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
